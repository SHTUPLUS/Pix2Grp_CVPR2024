# import os.path as osp
# import json


# import mmcv
# import numpy as np
# from PIL import Image


# CC3M_file_clnt = dict(
#     backend='petrel',
#     # path_mapping={
#     #     'cc3m': "s3://openmmlab/datasets/multimodal/cc3m",
#     #     './cc3m': "s3://openmmlab/datasets/multimodal/cc3m",
#     # }
# )


# class S3ImgReader():
#     def __init__(self, file_client_args,  data_prefix,
#                  color_type='color',
#                  channel_order='rgb',):
#         self.file_client = mmcv.FileClient(**file_client_args)
#         self.data_prefix = data_prefix
#         self.color_type = color_type
#         self.channel_order = channel_order

#     def get_image(self, filename):

#         url = osp.join(self.data_prefix, filename)
#         # print(url)
#         img_bytes = self.file_client.get(url)
#         img = mmcv.imfrombytes(
#             img_bytes,
#             flag=self.color_type,
#             channel_order=self.channel_order)
#         img = img.astype(np.uint8)
#         return Image.fromarray(img)



# if __name__ == '__main__':
#     with open("/mnt/lustre/lirongjie/tmp/cc3m_anno/val_annot.json", 'r') as f:
#         val_annot = json.load(f)

#     image_list = ['/'.join(each[0].split('/')[-4:]) for each in val_annot[:30]]
#     captions_list = [each[1].split('/') for each in val_annot[:30]]

#     s3_reader_test = S3ImgReader(CC3M_file_clnt, "s3://openmmlab/datasets/multimodal/cc3m")