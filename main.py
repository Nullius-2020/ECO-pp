import argparse
import os
import time
import shutil

import paddle.fluid.io as io
import paddle
import paddle.fluid as fluid
from dataset import TSNDataSet
from models import TSN
from collections import OrderedDict
from transforms import *
from opts import parser
import sys

best_prec1 = 0
os.environ['CPU_NUM']="2"

def main():
    global args, best_prec1
    args = parser.parse_args()
    

    print("------------------------------------")
    print("Environment Versions:")
    print("- Python: {}".format(sys.version))

    args_dict = args.__dict__
    print("------------------------------------")
    print(args.arch+" Configurations:")
    for key in args_dict.keys():
        print("- {}: {}".format(key, args_dict[key]))
    print("------------------------------------")

    if args.dataset == 'ucf101':
        num_class = 101
        rgb_read_format = "{:05d}.jpg"
    elif args.dataset == 'hmdb51':
        num_class = 51
        rgb_read_format = "{:05d}.jpg"
    elif args.dataset == 'kinetics':
        num_class = 400
        rgb_read_format = "{:04d}.jpg"
    elif args.dataset == 'something':
        num_class = 174
        rgb_read_format = "{:04d}.jpg"
    else:
        raise ValueError('Unknown dataset '+args.dataset)
    
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model = TSN(num_class, args.num_segments, args.pretrained_parts, args.modality,
                base_model=args.arch,
                consensus_type=args.consensus_type, dropout=args.dropout, partial_bn=not args.no_partialbn)

    crop_size = model.crop_size
    scale_size = model.scale_size
    input_mean = model.input_mean
    input_std = model.input_std
    
    # Optimizer s also support specifying per-parameter options. 
    # To do this, pass in an iterable of dict s. 
    # Each of them will define a separate parameter group, 
    # and should contain a params key, containing a list of parameters belonging to it. 
    # Other keys should match the keyword arguments accepted by the optimizers, 
    # and will be used as optimization options for this group.
    policies = model.get_optim_policies()

    train_augmentation = model.get_augmentation()

    

    model_dict = model.state_dict()

    
    if args.resume:
        if os.path.isfile(args.resume):
            place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
            with fluid.dygraph.guard(place):
                print(("=> loading checkpoint '{}'".format(args.resume)))
                para_state_dict, opti_state_dict = fluid.dygraph.load_dygraph(args.resume)
            # if not checkpoint['lr']:
            if "lr" not in  opti_state_dict.keys():
                args.lr = input("No 'lr' attribute found in resume model, please input the 'lr' manually: ")
                args.lr = float(args.lr)
            else:
                args.lr = opti_state_dict['lr']
            args.start_epoch = opti_state_dict['epoch']
            best_prec1 = opti_state_dict['best_prec1']
            model.load_state_dict(para_state_dict['state_dict'])
            print(("=> loaded checkpoint '{}' (epoch: {}, lr: {})"
                  .format(args.resume, opti_state_dict['epoch'], args.lr)))
        else:
            print(("=> no checkpoint found at '{}'".format(args.resume)))
    else:
         if args.arch == "ECO":
            new_state_dict = init_ECO(model_dict)
         elif args.arch == "ECOfull":
            new_state_dict = init_ECOfull(model_dict)
         elif args.arch == "C3DRes18":
            new_state_dict = init_C3DRes18(model_dict)


         
         un_init_dict_keys = [k for k in model_dict.keys() if k not in new_state_dict]
         print("un_init_dict_keys: ", un_init_dict_keys)
         print("\n------------------------------------")
         
         for k in un_init_dict_keys:
             new_state_dict[k] = fluid.layers.zeros(model_dict[k].shape, dtype='float64')
             if 'weight' in k:
                if 'bn' in k:
                   print("{} init as: 1".format(k))
                   new_state_dict[k]=fluid.layers.create_parameter(
                       shape=new_state_dict[k].shape,
                       dtype='float64',
                       name='bn1',
                       attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.Constant(value=1.0)))
                else:
                   print("{} init as: xavier".format(k))
                   new_state_dict[k]=fluid.layers.create_parameter(
                       shape=new_state_dict[k].shape,
                       dtype='float64',
                       name='bn2',
                       attr=fluid.param_attr.ParamAttr(
                                  initializer=ffluid.initializer.XavierInitializer(uniform=True,fan_in=None,fan_out=None,seed=0)))
                   
             elif 'bias' in k:
                  print("{} init as: 0".format(k))
                  new_state_dict[k]=fluid.layers.create_parameter(
                       shape=new_state_dict[k].shape,
                       dtype='float64',
                       name='bn3',
                       attr=fluid.param_attr.ParamAttr(
                                  initializer=fluid.initializer.Constant(value=0)))

         print("------------------------------------")
         
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        model.set_dict(new_state_dict)
        



     # Data loading code
    if args.modality != 'RGBDiff':
        #input_mean = [0,0,0] #for debugging
        normalize = GroupNormalize(input_mean, input_std)
    else:
        normalize = IdentityTransform()
        

    if args.modality == 'RGB':
        data_length = 1
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    #print(args.train_list)
    fset=TSNDataSet("", args.train_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl='img_'+rgb_read_format if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+rgb_read_format,
                   transform=Compose([
                       #GroupScale(int(scale_size)),
                       train_augmentation,
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       normalize,
                   ]))

    train_loader=fset
    vfset=TSNDataSet("", args.val_list, num_segments=args.num_segments,
                   new_length=data_length,
                   modality=args.modality,
                   image_tmpl='img_'+rgb_read_format if args.modality in ["RGB", "RGBDiff"] else args.flow_prefix+rgb_read_format,
                   random_shift=False,
                   transform=Compose([
                       GroupScale(int(scale_size)),
                       GroupCenterCrop(crop_size),
                       Stack(roll=True),
                       ToTorchFormatTensor(div=False),
                       #Stack(roll=(args.arch == 'C3DRes18') or (args.arch == 'ECO') or (args.arch == 'ECOfull') or (args.arch == 'ECO_2FC')),
                       #ToTorchFormatTensor(div=(args.arch != 'C3DRes18') and (args.arch != 'ECO') and (args.arch != 'ECOfull') and (args.arch != 'ECO_2FC')),
                       normalize,
                   ]))
    val_loader=vfset
  

   
    # define loss function (criterion) and optimizer
    if args.loss_type == 'nll':
        #criterion = torch.nn.CrossEntropyLoss().cuda()
        criterion = 'cross_entropy'
    else:
        raise ValueError("Unknown loss type")
    
    for group in policies:
        print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
            group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):

        clip = fluid.clip.GradientClipByNorm(clip_norm=args.clip_gradient)
        optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=args.lr,\
        momentum=args.momentum,regularization=fluid.regularizer.L2Decay(args.weight_decay),\
        use_nesterov=args.nesterov,grad_clip=clip,parameter_list=model.parameters())

        if args.evaluate:
            validate(val_loader, model, criterion, 0,arg.batch_size)
            return

    saturate_cnt = 0
    exp_num = 0

    for epoch in range(args.start_epoch, args.epochs):

        if saturate_cnt == args.num_saturate:
            exp_num = exp_num + 1
            saturate_cnt = 0
            print("- Learning rate decreases by a factor of '{}'".format(10**(exp_num)))
        #adjust_learning_rate(optimizer, epoch, args.lr_steps, exp_num)
            decay = 0.1 ** (exp_num)
            lr = args.lr * decay
            decay = args.weight_decay
 
            with fluid.dygraph.guard(place):
                clip = fluid.clip.GradientClipByNorm(clip_norm=args.clip_gradient)
                optimizer = fluid.optimizer.MomentumOptimizer(learning_rate=lr,\
                    momentum=args.momentum,regularization=fluid.regularizer.L2Decay(decay),\
                    use_nesterov=args.nesterov,grad_clip=clip,parameter_list=model.parameters())

        
    
        place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
        with fluid.dygraph.guard(place):

   
            train(train_loader, model, criterion, optimizer, epoch,args.batch_size)

        # evaluate on validation set
            if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
                prec1 = validate(val_loader, model, criterion, (epoch + 1) * len(train_loader),0,args.batch_size)

            # remember best prec@1 and save checkpoint
                is_best = prec1 > best_prec1
                if is_best:
                    saturate_cnt = 0
                else:
                    saturate_cnt = saturate_cnt + 1

                print("- Validation Prec@1 saturates for {} epochs.".format(saturate_cnt))
                best_prec1 = max(prec1, best_prec1)
                save_checkpoint(model.state_dict(), is_best)
                '''
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'lr': optimizer.current_step_lr(),
                }, is_best)
                '''
def init_ECO(model_dict):

    weight_url_2d='https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'

    if args.pretrained_parts == "scratch":
            
        new_state_dict = {}

    elif args.pretrained_parts == "2D":

        if args.net_model2D is not None:
            pretrained_dict_2d = fluid.dygraph.load_dygraph(args.net_model2D)
            print(("=> loading model - 2D net:  '{}'".format(args.net_model2D)))
        else:
            weight_url_2d='https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'
            pretrained_dict_2d = fluid.dygraph.load_dygraph(weight_url_2d)
            print(("=> loading model - 2D net-url:  '{}'".format(weight_url_2d)))

        #print(pretrained_dict_2d)
        for k, v in pretrained_dict_2d['state_dict'].items():
             if "module.base_model."+k in model_dict:
                 print("k is in model dict", k)
             else:
                 print("Problem!")
                 print("k: {}, size: {}".format(k,v.shape))
       
        new_state_dict = {"module.base_model."+k: v for k, v in pretrained_dict_2d['state_dict'].items() if "module.base_model."+k in model_dict}

    elif args.pretrained_parts == "3D":

        new_state_dict = {}
        if args.net_model3D is not None:
            pretrained_dict_3d = fluid.dygraph.load_dygraph(args.net_model3D)
            print(("=> loading model - 3D net:  '{}'".format(args.net_model3D)))
        else:
            pretrained_dict_3d = fluid.dygraph.load_dygraph("models/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
            print(("=> loading model - 3D net-url:  '{}'".format("models/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")))

        for k, v in pretrained_dict_3d['state_dict'].items():
            if (k in model_dict) and (v.size() == model_dict[k].size()):
                new_state_dict[k] = v

        res3a_2_weight_chunk = torch.chunk(pretrained_dict_3d["state_dict"]["module.base_model.res3a_2.weight"], 4, 1)
        new_state_dict["module.base_model.res3a_2.weight"] = tensor.concat((res3a_2_weight_chunk[0], res3a_2_weight_chunk[1], res3a_2_weight_chunk[2]), 1)


    elif args.pretrained_parts == "finetune":
        print(args.net_modelECO)
        print("88"*40)
        if args.net_modelECO is not None:
            pretrained_dict = fluid.dygraph.load_dygraph(args.net_modelECO)
            print(("=> loading model-finetune: '{}'".format(args.net_modelECO)))
        else:
            pretrained_dict = fluid.dygraph.load_dygraph("models/eco_lite_rgb_16F_kinetics_v2.pth.tar")
            print(("=> loading model-finetune-url: '{}'".format("models/eco_lite_rgb_16F_kinetics_v2.pth.tar")))




        new_state_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if (k in model_dict) and (v.size() == model_dict[k].size())}
        print("*"*50)
        print("Start finetuning ..")

    elif args.pretrained_parts == "both":

        # Load the 2D net pretrained model
        if args.net_model2D is not None:
            pretrained_dict_2d = fluid.dygraph.load_dygraph(args.net_model2D)
            print(("=> loading model - 2D net:  '{}'".format(args.net_model2D)))
        else:
            weight_url_2d='https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'
            pretrained_dict_2d = fluid.dygraph.load_dygraph(weight_url_2d)
            print(("=> loading model - 2D net-url:  '{}'".format(weight_url_2d)))


        # Load the 3D net pretrained model
        if args.net_model3D is not None:
            pretrained_dict_3d = fluid.dygraph.load_dygraph(args.net_model3D)
            print(("=> loading model - 3D net:  '{}'".format(args.net_model3D)))
        else:
            pretrained_dict_3d = fluid.dygraph.load_dygraph("models/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
            print(("=> loading model - 3D net-url:  '{}'".format("models/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")))

        new_state_dict = {"module.base_model."+k: v for k, v in pretrained_dict_2d['state_dict'].items() if "module.base_model."+k in model_dict}

        for k, v in pretrained_dict_3d['state_dict'].items():
            if (k in model_dict) and (v.size() == model_dict[k].size()):
                new_state_dict[k] = v

        res3a_2_weight_chunk = torch.chunk(pretrained_dict_3d["state_dict"]["module.base_model.res3a_2.weight"], 4, 1)
        new_state_dict["module.base_model.res3a_2.weight"] = torch.cat((res3a_2_weight_chunk[0], res3a_2_weight_chunk[1], res3a_2_weight_chunk[2]), 1)
    return new_state_dict

def init_ECOfull(model_dict):

    weight_url_2d='https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'

    if args.pretrained_parts == "scratch":
            
        new_state_dict = {}

    elif args.pretrained_parts == "2D":

        pretrained_dict_2d = fluid.dygraph.load_dygraph(weight_url_2d)
        new_state_dict = {"module.base_model."+k: v for k, v in pretrained_dict_2d['state_dict'].items() if "module.base_model."+k in model_dict}

    elif args.pretrained_parts == "3D":

        new_state_dict = {}
        pretrained_dict_3d = fluid.dygraph.load_dygraph("models/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
        for k, v in pretrained_dict_3d['state_dict'].items():
            if (k in model_dict) and (v.size() == model_dict[k].size()):
                new_state_dict[k] = v

        res3a_2_weight_chunk = fluid.layers.crop_tensor(pretrained_dict_3d["state_dict"]["module.base_model.res3a_2.weight"], 4, 1)
        new_state_dict["module.base_model.res3a_2.weight"] = tensor.concat((res3a_2_weight_chunk[0], res3a_2_weight_chunk[1], res3a_2_weight_chunk[2]), 1)



    elif args.pretrained_parts == "finetune":
        print(args.net_modelECO)
        print("88"*40)
        
        if args.net_modelECO is not None:
            with fluid.dygraph.guard():
                pretrained_dict,_ = fluid.dygraph.load_dygraph(args.net_modelECO)
            print(("=> loading model-finetune: '{}'".format(args.net_modelECO)))
        else:
            pretrained_dict = fluid.dygraph.load_dygraph("models/eco_lite_rgb_16F_kinetics_v2.pth.tar")
            print(("=> loading model-finetune-url: '{}'".format("models/eco_lite_rgb_16F_kinetics_v2.pth.tar")))
        
        new_state_dict=OrderedDict()
        for m_key in model_dict.keys():
            if not (m_key.find('fc')):
                new_state_dict[m_key]=pretrained_dict[m_key]
            else:
                new_state_dict[m_key]=model_dict[m_key]
            print(m_key,"\t",model_dict[m_key].shape, new_state_dict[m_key].shape)
        print("*"*50)
        print("Start finetuning ..")

    elif args.pretrained_parts == "both":

        # Load the 2D net pretrained model
        if args.net_model2D is not None:
            pretrained_dict_2d = fluid.dygraph.load_dygraph(args.net_model2D)
            print(("=> loading model - 2D net:  '{}'".format(args.net_model2D)))
        else:
            weight_url_2d='https://yjxiong.blob.core.windows.net/ssn-models/bninception_rgb_kinetics_init-d4ee618d3399.pth'
            pretrained_dict_2d = fluid.dygraph.load_dygraph(weight_url_2d)
            print(("=> loading model - 2D net-url:  '{}'".format(weight_url_2d)))

        new_state_dict = {"module.base_model."+k: v for k, v in pretrained_dict_2d['state_dict'].items() if "module.base_model."+k in model_dict}

        # Load the 3D net pretrained model
        if args.net_model3D is not None:
            pretrained_dict_3d = fluid.dygraph.load_dygraph(args.net_model3D)
            print(("=> loading model - 3D net:  '{}'".format(args.net_model3D)))
        else:
            pretrained_dict_3d = fluid.dygraph.load_dygraph("models/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
            print(("=> loading model - 3D net-url:  '{}'".format("models/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")))

        
        for k, v in pretrained_dict_3d['state_dict'].items():
            if (k in model_dict) and (v.size() == model_dict[k].size()):
                new_state_dict[k] = v
        

    return new_state_dict

def init_C3DRes18(model_dict):

    if args.pretrained_parts == "scratch":
        new_state_dict = {}
    elif args.pretrained_parts == "3D":
        pretrained_dict = fluid.dygraph.load_dygraph("models/C3DResNet18_rgb_16F_kinetics_v1.pth.tar")
        new_state_dict = {k: v for k, v in pretrained_dict['state_dict'].items() if (k in model_dict) and (v.size() == model_dict[k].size())}
    else:
        raise ValueError('For C3DRes18, "--pretrained_parts" can only be chosen from [scratch, 3D]')

    return new_state_dict



def train(train_loader, model, criterion, optimizer, epoch,batch_size):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    

    if args.no_partialbn:
        model.partialBN(False)
    else:
        model.partialBN(True)

    

    # switch to train mode
    model.train()

    end = time.time()

    loss_summ = 0
    localtime   = time.localtime()
    end_time  = time.strftime("%Y/%m/%d-%H:%M:%S", localtime)
    batch=len(train_loader)//batch_size
    
    for i in range (batch):

        image,label =train_loader.__getitem__(i)
        image = np.array(image).astype('float32').reshape(-1,3,224,224)
        label = np.array(label).astype('int64').reshape(-1,1)

        inputs = fluid.dygraph.to_variable(image)
        target= fluid.dygraph.to_variable(label)
        target.stop_gradient = True



        
        # measure data loading time
        data_time.update(time.time() - end)

        inputs  = fluid.dygraph.to_variable(inputs)
            
        input_var = inputs
        target_var = target
        

        output = model(input_var)

        loss = fluid.layers.softmax_with_cross_entropy(output, target_var)

        loss =fluid.layers.mean(loss)

        loss_summ += loss.numpy()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1,5))
        losses.update(loss_summ[0],inputs.shape[0])
        top1.update(prec1[0],inputs.shape[0])
        top5.update(prec5[0], inputs.shape[0])

        loss.backward()
        optimizer.minimize(loss)
        model.clear_gradients()


        if (i+1) % args.iter_size == 0:
            # scale down gradients when iter size is functioning

            
            loss_summ = 0
            lr=optimizer.current_step_lr()

            print(('Epoch: [{0}][{1}/{2}], lr: {lr:.7f}\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'UTime {end_time:} \t'
                  'Data {data_time.val:.2f} ({data_time.avg:.2f})\t'
                  'Loss {losses.val:.3f} ({losses.avg:.3f})\t'
                  'Prec@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  'Prec@5 {top5.val:.2f} ({top5.avg:.2f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,end_time=end_time,
                   data_time=data_time, losses=losses, top1=top1, top5=top5,lr=lr)))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        localtime   = time.localtime()
        end_time  = time.strftime("%Y/%m/%d-%H:%M:%S", localtime)




def validate(val_loader, model, criterion, iter,logger,batch_size):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()



    # switch to evaluate mode
    model.eval()

    end = time.time()
    batch=len(val_loader)//batch_size
    
    for i in range (batch):

        image,label =val_loader.__getitem__(i)
        image = np.array(image).astype('float32').reshape(-1,3,224,224)
        label = np.array(label).astype('int64').reshape(-1,1)
        

        inputs = fluid.dygraph.to_variable(image)
        target= fluid.dygraph.to_variable(label)
        target.stop_gradient = True
        input_var = inputs
        target_var = target

        # compute output
        output = model(input_var)
       
        loss = fluid.layers.softmax_with_cross_entropy(output, target_var)

        # measure acprrincuracy and record loss
        prec1, prec5 = accuracy(output, target, topk=(1,5))

        loss = fluid.layers.mean(loss)
        losses.update(loss.numpy()[0], inputs.shape[0])
        top1.update(prec1[0], inputs.shape[0])
        top5.update(prec5[0], inputs.shape[0])
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print(('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, losses=losses,
                   top1=top1, top5=top5)))

    print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
          .format(top1=top1, top5=top5, loss=losses)))

    return top1.avg


def save_checkpoint(state, is_best, filename='net_runs/ECO-pp'):

    if is_best:
        paddle.fluid.dygraph.save_dygraph(state, filename)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, lr_steps, exp_num):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #This function is replaced in the main loop of epoch.
    # decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    decay = 0.1 ** (exp_num)
    lr = args.lr * decay
    decay = args.weight_decay
    
    #for param_group in optimizer.parameters():
    #    param_group['lr'] = lr * param_group['lr_mult']
    #    param_group['weight_decay'] = decay * param_group['decay_mult']
  
    

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    
    batch_size = target.shape[0]
    prec1 = fluid.layers.accuracy(input=output, label=target, k=1)
    prec5 = fluid.layers.accuracy(input=output, label=target, k=5)
    prec1=prec1.numpy()*100
    prec5=prec5.numpy()*100
    return prec1,prec5
    



if __name__ == '__main__':
    main()
