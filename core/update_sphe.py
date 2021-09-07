import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def func_conv_deform(x, loc_layer, k, s, layers_act_num, offset_file = '', activated = False):
    # print(layers_act_num)
    if offset_file == '':
        offset_file = './OFFSETS/offset_'+str(int(x.shape[3]/s))+'_'+str(int(x.shape[2]/s))+'_'+str(k)+'_'+str(k)+'_'+str(s)+'_'+str(s)+'_1'+'.pt'
    if activated and layers_act_num <= 400:
        print(layers_act_num, " activated")
        offset = torch.load(offset_file).cuda()
        if x.shape[0] != 1: 
            offset = torch.cat([offset for _ in range(x.shape[0])],dim=0)
    else:
        print(layers_act_num, " not activated")
        offset = torch.zeros(x.shape[0],2*k*k,int(x.shape[2]/s),int(x.shape[3]/s)).cuda()
    offset.require_gradient = False
    y = loc_layer(x,offset)
    del offset
    torch.cuda.empty_cache()
    return y

def func_conv_deform_2(x, loc_layer, kw, kh, sw, sh, layers_act_num, offset_file = '', activated = False):
    # print(layers_act_num)
    if offset_file == '':
        offset_file = './OFFSETS/offset_'+str(int(x.shape[3]/sw))+'_'+str(int(x.shape[2]/sh))+'_'+str(kw)+'_'+str(kh)+'_'+str(sw)+'_'+str(sh)+'_1'+'.pt'
    if activated and layers_act_num <= 400:
        print(layers_act_num, " activated")
        offset = torch.load(offset_file).cuda()
        # print(offset)
        if x.shape[0] != 1: 
            offset = torch.cat([offset for _ in range(x.shape[0])],dim=0)
    else:
        print(layers_act_num, " not activated")
        offset = torch.zeros(x.shape[0],2*kw*kh,int(x.shape[2]/sw),int(x.shape[3]/sh)).cuda()
    offset.require_gradient = False
    y = loc_layer(x,offset)
    del offset
    torch.cuda.empty_cache()
    return y


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        # self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv1 = torchvision.ops.DeformConv2d(input_dim, hidden_dim, 3, padding=1)
        # self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.conv2 = torchvision.ops.DeformConv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.relu(func_conv_deform(x, self.conv1, 3, 1, 221, '', False))
        return func_conv_deform(y, self.conv2, 3, 1, 222, '', False)

class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h

class SepConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        # self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        # self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        # self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convz1 = torchvision.ops.DeformConv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = torchvision.ops.DeformConv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = torchvision.ops.DeformConv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        # self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        # self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        # self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

        self.convz2 = torchvision.ops.DeformConv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = torchvision.ops.DeformConv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = torchvision.ops.DeformConv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        
        
    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        # z = torch.sigmoid(self.convz1(hx))
        z = torch.sigmoid(func_conv_deform_2(hx, self.convz1, 5, 1, 1, 1, 211,'', False))
        # r = torch.sigmoid(self.convr1(hx))
        r = torch.sigmoid(func_conv_deform_2(hx, self.convr1, 5, 1, 1, 1, 212,'', False))
        # q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))  
        q = torch.tanh(func_conv_deform_2(torch.cat([r*h, x], dim=1), self.convq1, 5, 1, 1, 1, 213,'', False))
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        # z = torch.sigmoid(self.convz2(hx))
        z = torch.sigmoid(func_conv_deform_2(hx, self.convz2, 1, 5, 1, 1, 214,'', False)) 
        # r = torch.sigmoid(self.convr2(hx))
        r = torch.sigmoid(func_conv_deform_2(hx, self.convr2, 1, 5, 1, 1, 215,'', False))
        # q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))  
        q = torch.tanh(func_conv_deform_2(torch.cat([r*h, x], dim=1), self.convq2, 1, 5, 1, 1, 216,'', False))
        h = (1-z) * h + z * q

        return h

class SmallMotionEncoder(nn.Module):
    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)

class BasicMotionEncoder(nn.Module):
    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        # self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convc2 = torchvision.ops.DeformConv2d(256, 192, 3, padding=1)
        # self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf1 = torchvision.ops.DeformConv2d(2, 128, 7, padding=3)
        # self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.convf2 = torchvision.ops.DeformConv2d(128, 64, 3, padding=1)
        # self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)
        self.conv = torchvision.ops.DeformConv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc1(corr))
        # cor = F.relu(self.convc2(cor))
        cor = F.relu(func_conv_deform(cor, self.convc2, 3, 1, 201, '', False))
        # flo = F.relu(self.convf1(flow))
        flo = F.relu(func_conv_deform(flow, self.convf1, 7, 1, 202, '', False))
        # flo = F.relu(self.convf2(flo))
        flo = F.relu(func_conv_deform(flo, self.convf2, 3, 1, 203, '', False))

        cor_flo = torch.cat([cor, flo], dim=1)
        # out = F.relu(self.conv(cor_flo))
        out = F.relu(func_conv_deform(cor_flo, self.conv, 3, 1, 204, '', False))
        return torch.cat([out, flow], dim=1)

class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow

class mySequential(nn.Sequential):
    def forward(self, *input):
        for module in self._modules.values():
            if type(input) == tuple:
                input = module(*input)
            else:
                input = module(input)
        return input

class BasicUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=128, input_dim=128):
        super(BasicUpdateBlock, self).__init__()
        self.args = args
        self.encoder = BasicMotionEncoder(args)
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        # self.mask = nn.Sequential(
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 64*9, 1, padding=0))

        self.mask = mySequential(
            torchvision.ops.DeformConv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))


    def forward(self, net, inp, corr, flow, upsample=True, num_l_d=0):
        # print(num_l_d)
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)

        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        # scale mask to balence gradients
        activated_231 = False
        if activated_231 == True:
            print("231 activated")
            offset_file = './OFFSETS/offset_'+str(int(net.shape[3]/1))+'_'+str(int(net.shape[2]/1))+'_3_3_1_1_1'+'.pt'
            offset = torch.load(offset_file).cuda()
        else:
            print("231 not activated")
            offset = torch.zeros(net.shape[0],2*3*3,int(net.shape[2]/1),int(net.shape[3]/1)).cuda()
        offset.require_gradient = False
        
        mask = .25 * self.mask(net, offset)
        del offset
        torch.cuda.empty_cache()
        return net, mask, delta_flow



