#
# Copyright 2015 By ihciah
# https://github.com/ihciah/CNN_forward
#
import caffe,struct
import numpy as np
#from PIL import Image


class Convertor:
    def __init__(self,output,net):
        self.output=open(output,"wb+",0)
        self.net=net

    def write_data(self,layer_id,layer_name):
        d=self.net.layers[layer_id]
        if d.type=='Convolution':
            shape=list(d.blobs[0].data.shape) #conv
            allcount=1
            for i in shape:
                allcount*=i
            layer_name=list(layer_name[:15])
            for i in range(16-len(layer_name)):
                layer_name.append("\0")
            type='c'
            print layer_name
            self.output.write(struct.pack("16c", *layer_name))
            self.output.write(struct.pack("c", *type))
            self.output.write(struct.pack("4i", *shape))
            self.output.write(struct.pack("i", allcount))
            self.output.write(struct.pack("%sf" %allcount, *d.blobs[0].data.flatten()))

            shape=list(d.blobs[1].data.shape) #bias
            allcount=1
            for i in shape:
                allcount*=i

            self.output.write(struct.pack("i", allcount))
            self.output.write(struct.pack("%sf" %allcount, *d.blobs[1].data.flatten()))
            self.output.write(struct.pack("i",0))
        if d.type=='InnerProduct':
            shape=list(d.blobs[0].data.shape) #dense
            allcount=1
            for i in shape:
                allcount*=i
            layer_name=list(layer_name[:15])
            for i in range(16-len(layer_name)):
                layer_name.append("\0")
            type='d'
            print layer_name
            self.output.write(struct.pack("16c", *layer_name))
            self.output.write(struct.pack("c", *type))
            self.output.write(struct.pack("2i", *shape))
            self.output.write(struct.pack("i", 1))
            self.output.write(struct.pack("i", 1))
            self.output.write(struct.pack("i", allcount))
            self.output.write(struct.pack("%sf" %allcount, *d.blobs[0].data.flatten()))

            shape=list(d.blobs[1].data.shape) #bias
            allcount=1
            for i in shape:
                allcount*=i

            self.output.write(struct.pack("i", allcount))
            self.output.write(struct.pack("%sf" %allcount, *d.blobs[1].data.flatten()))
            self.output.write(struct.pack("i",0))


if __name__=="__main__":
    DEPLOY_PROTOTXT="/Users/weizheliu/Desktop/nViso/Caffe-mini/50x50_arch/deploy_trained_model.prototxt"
    TRAINED_NET="/Users/weizheliu/Desktop/nViso/Caffe-mini/50x50_arch/trained_model_s4_iter_1317120.caffemodel"
    OUTPUT_PATH = '/Users/weizheliu/Desktop/nViso/Caffe-mini/50x50_arch/nViso_model'

    net = caffe.Classifier(DEPLOY_PROTOTXT,TRAINED_NET)
    print net
    print list(net._layer_names)
    print net.layers[0].blobs[0].data.shape
    conv=Convertor(OUTPUT_PATH,net)
    conv.write_data(0,"conv0_1")
    conv.write_data(2,"conv0_2")
    conv.write_data(5,"conv1_1")
    conv.write_data(7,"conv1_2")
    conv.write_data(10,"conv2_1")
    conv.write_data(12,"conv2_2")
    conv.write_data(15,"conv3_1")
    conv.write_data(17,"conv3_2")
    conv.write_data(21,"ip0_1")
    conv.write_data(23,"ip0_2")
    conv.write_data(24,"ip1_1")
    conv.write_data(26,"ip1_2")
    conv.write_data(27,"ip2_1")
    conv.write_data(29,"ip2_2")
    conv.write_data(30,"ip3_1")
    conv.write_data(32,"ip3_2")
    conv.write_data(33,"ip4_1")
    conv.write_data(35,"ip4_2")
    conv.write_data(36,"ip5_1")
    conv.write_data(38,"ip5_2")
    conv.write_data(39,"ip6_1")
    conv.write_data(41,"ip6_2")
    conv.write_data(42,"ip7_1")
    conv.write_data(44,"ip7_2")
    conv.write_data(45,"ip8_1")
    conv.write_data(47,"ip8_2")
