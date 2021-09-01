import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import socket  
import numpy  
import cv2  
import time

def detect_img(yolo):
    """
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image, sum, class_names = yolo.detect_image(image)
            print(str(sum) + '個見つかりました')
            for i in range(sum):
                print(class_names[i] + 'が見つかりました')
            r_image.show()
    """
    image = Image.open("sample.jpg")

    """
    r_image : 解析後の画像
    sum : 解析して見つかった数の個数
    class_names : 見つかった数(リスト)
    """
    r_image, sum, class_names = yolo.detect_image(image)
    print(str(sum) + '個見つかりました')
    for i in range(sum):
        print(class_names[i] + 'が見つかりました')
    r_image.show()

    #yolo.close_session()

FLAGS = None

def photo():
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")

def getimage():
    #socketをつかってラズパイと解析用PCをつなぐ
    #IPアドレスとポート番号は環境に応じて変更
    HOST = '192.168.11.100'
    PORT = 54328
    sock=socket.socket(socket.AF_INET,socket.SOCK_STREAM)  
    sock.connect((HOST,PORT))   
    
    buf=b''
    recvlen=4096  
    
    #受け取るデータはstr型なのでimageに変換
    while recvlen == 4096:  
        receivedstr=sock.recv(4096)
        recvlen=len(receivedstr)  
        buf +=receivedstr
    buf += b''
    sock.close()  
    narray=numpy.fromstring(buf,dtype='uint8')  
    return cv2.imdecode(narray,1)  

if __name__ == '__main__':
    while True:  
        img = getimage()
        #cv2.imshow('Capture',img) 
        #cv2.waitKey(0)
        cv2.imwrite("sample.jpg", img)
        photo()

    HOST = '192.168.11.100'
    PORT = 54328
    getimage()
        