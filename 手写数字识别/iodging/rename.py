# -*- coding: utf-8 -*-
# @Time : 2021/9/13 16:21
# @Author : 黄小渣
# @FileName: rename.py
# @Software: PyCharm
import os

class BatchRename():
    def __init__(self):
        self.path = 'C:/Users/Administrator/Desktop/倒伏/heji'
    def rename(self):
        filelist = os.listdir(self.path)
        total_sum = len(filelist)
        i = 1
        for item in filelist:
            if item.endswith('.jpg') or item.endswith('.png') or item.endswith('.jpeg'):
                src = os.path.join(os.path.abspath(self.path),item)
                dst = os.path.join(os.path.abspath(self.path),format(str(i),'0>4s') + '.jpg')
                try :
                    os.rename(src,dst)
                    print('converting {} to {} ...'.format(src,dst))
                    print('进度完成了{}/{}'.format(i, total_sum))
                    i = i + 1
                except:
                    continue
        print('total {} to rename & converted {} jpgs'.format(total_sum,i))


if __name__ == '__main__':
    demo = BatchRename()
    demo.rename()

