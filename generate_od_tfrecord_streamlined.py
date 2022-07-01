"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import io
import cv2, glob, os, json
import shutil
import pandas as pd
import tensorflow as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
flags.DEFINE_float('image_size_modifier', 0.25, 'Image size scale down')
flags.DEFINE_float('train_images_fraction', 0.85, 'Fraction of images to be used for training, rest for validation')
FLAGS = flags.FLAGS

def utility():
    csv_inp = FLAGS.csv_input #/media/photon/5290fe77-c5d4-4e12-96f2-30b906f296cd/Code/verizon/training_sets/test/
    csv_op = os.path.join(csv_inp, 'Master_CSV', 'Final.csv')
    img_master = FLAGS.image_dir #/media/photon/5290fe77-c5d4-4e12-96f2-30b906f296cd/Code/verizon/data/video_frames/
    resz_temp = os.path.join(csv_inp, "Resized_temp")
    trainval_ratio = FLAGS.train_images_fraction
    image_size_modifier = FLAGS.image_size_modifier

    final_df = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

    if not os.path.isdir(resz_temp):
        os.mkdir(resz_temp)

    for csv in glob.glob(os.path.join(csv_inp, "*.csv")):

        df = pd.read_csv(csv)

        df = df.dropna()
        df = df[df.region_count != 0]

        df['bbox_json'] = df['region_shape_attributes'].apply(json.loads)
        df['frame_no'] = df['filename'].apply(lambda x: int(x.split('_')[-1].split('.')[0]))

        df['x'] = df['bbox_json'].apply(lambda x: x['x'])
        df['y'] = df['bbox_json'].apply(lambda x: x['y'])
        df['width'] = df['bbox_json'].apply(lambda x: x['width'])
        df['height'] = df['bbox_json'].apply(lambda x: x['height'])

        #resizing images
        folder = df.values[0][0].split('_')[1] + "_" + df.values[0][0].split('_')[2]
        csv_image_path = os.path.join(img_master, folder, "full_images")
        crop_image_path = os.path.join(img_master, folder, "cropped_images")
        if not os.path.isdir(crop_image_path):
            os.mkdir(crop_image_path)

        for inx, row in df.iterrows():

            img = cv2.imread(os.path.join(csv_image_path, row.filename))

            if img is not None:
                crop = img[row.y:row.y+row.height, row.x:row.x+row.width, :]
                img = cv2.resize(img, dsize=(0, 0), fx=image_size_modifier, fy=image_size_modifier)
                cv2.imwrite(os.path.join(crop_image_path, row[0]), crop)
                cv2.imwrite(os.path.join(resz_temp, row.filename), img)
            else:
                print("Image not found: "+ csv_image_path+row[0])

        df['x'] = (image_size_modifier * df['x'].apply(int)).apply(int)
        df['y'] = (image_size_modifier * df['y'].apply(int)).apply(int)
        df['width'] = (image_size_modifier * df['width'].apply(int)).apply(int)
        df['height'] = (image_size_modifier * df['height'].apply(int)).apply(int)

        #new_df = df['filename']
        new_df = pd.DataFrame(columns=['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        new_df['filename'] = df['filename']
        new_df['width'] = df['width']
        new_df['height'] = df['height']
        new_df['class'] = 'tower_head'
        new_df['xmin'] = df['x']
        new_df['ymin'] = df['y']
        new_df['xmax'] = new_df['xmin'] + new_df['width']
        new_df['ymax'] = new_df['ymin'] + new_df['height']

        #print("BEFORE APPEND")
        final_df = final_df.append(new_df)

        #print("APPENDED")
        # modify annotations based on condition
        # condition = new_df['filename'].str.contains('0482|0486')
        # new_df.loc[condition, ['xmin']] =  new_df[condition]['xmin'] - 50
        # new_df.loc[condition, ['xmax']] =  new_df[condition]['xmax'] + 50
        # new_df.loc[condition, ['ymin']] =  new_df[condition]['ymin'] - 50
        # new_df.loc[condition, ['ymax']] =  new_df[condition]['ymax'] + 50
        # new_df.loc[condition, ['width']] =  new_df[condition]['width'] + 100
        # new_df.loc[condition, ['height']] =  new_df[condition]['height'] + 100

        print(folder+" Video Annotations Picked up")
        print(str(new_df.shape)+" : No of annotations of current video")
        print(str(final_df.shape)+" : Total no of annotations")

    final_df = final_df.sample(frac=1)
    split_idx = int(final_df.shape[0] * trainval_ratio)
    train_df = final_df.iloc[:split_idx, :]
    val_df = final_df.iloc[split_idx:, :]

    if not os.path.isdir(os.path.join(csv_inp, "Master_CSV")):
        os.mkdir(os.path.join(csv_inp, "Master_CSV"))
    csv_train_path = csv_op[:-4] + "_train.csv"
    csv_val_path = csv_op[:-4] + "_val.csv"
    train_df.to_csv(csv_train_path, index=False)
    val_df.to_csv(csv_val_path, index=False)

    print("Total Annotations: ", final_df.shape, "Train split: ", train_df.shape, "Val Split: ", val_df.shape)

    return csv_train_path, csv_val_path, resz_temp

# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'tower_head':
        return 1
    else:
        None


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
        # print(xmins[-1], xmaxs[-1], ymins[-1], ymaxs[-1])
    assert max(xmins) < 1 and max(xmaxs) < 1 and max(ymins) < 1 and max(ymaxs) < 1
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_save_final_tf_record(out_path, resized_img_path, csv_path):
    writer = tf.python_io.TFRecordWriter(out_path)
    #path = os.path.join(FLAGS.image_dir)
    path=resized_img_path
    # examples = pd.read_csv(FLAGS.csv_input)
    examples = pd.read_csv(csv_path)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    out_path = os.path.join(os.getcwd(),out_path)
    print('Successfully created the TFRecords: {}'.format(out_path))


def main(_):

    train_csv_path, val_csv_path, resized_img_path = utility()
    print("CSVs combined, split and resized images dumped successfully!!")
    create_save_final_tf_record(os.path.join(FLAGS.output_path, "train.record"), resized_img_path, train_csv_path)
    create_save_final_tf_record(os.path.join(FLAGS.output_path, "val.record"), resized_img_path, val_csv_path)
    if os.path.isdir(resized_img_path):
        shutil.rmtree(resized_img_path)  
    print("Temporary Resized Images Deleted!!")


if __name__ == '__main__':
    tf.app.run()
