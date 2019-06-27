import argparse
from utils import *
import tensorflow as tf
import GAN_Model as model

def parse_args():
    desc = "Hair structure recovered from 2D Img Info."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('--train_dir', type=str,
                        default='../TrainData_Ori/',
                        help='dir of training data.')
    parser.add_argument('--test_dir', type=str,
                        default='../TestData_Ori/',
                        help='dir of testing data.')
    parser.add_argument('--If_train', type=bool,
                        default=False,
                        help='True: process training. False: process testing.')
    parser.add_argument('--batchSize', type=int,
                        default=4,
                        help='Batch size set for training process.')
    parser.add_argument('--rst_dir', type=str,
                        default='../rstData_LossComb/',
                        help='dir to save result.')
    parser.add_argument('--In_dir', type=str,
                        default='../Hairs/combine/comb_2/',
                        help='dir of Input data not for training.')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        myModel = model.Model(sess, trainRoot=args.train_dir, testRoot=args.test_dir, rstRoot=args.rst_dir,
                              batchSize=args.batchSize)

        if args.If_train:
            myModel.train(2000)
        else:
            myModel.test(args.In_dir)
            #myModel.interHairTest(args.In_dir, args.In_dir + "A/", args.In_dir + "B/")

if __name__ == '__main__':
    main()


