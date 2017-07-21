# 选取一部分反共图片和一部分非反共图片对训练出的模型就行评估
# 评估标准：
# 1. 准确率（包括各个类别的准确率）
# 2. 召回率（包括各个类别的召回率）
# 3. F1 Score
# =============================================================================

import tensorflow as tf
import glob

F_THRESHOLD = 0.6


def load_image(filename):
    """Read in the image_data to be classified."""
    return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
    """Read in labels, one label per line."""
    return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
    """Unpersists graph from file as default graph."""
    with tf.gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name='')


def run_graph(image_data, labels, input_layer_name='DecodeJpeg/contents:0',
              output_layer_name='final_result:0', num_top_predictions=5):
    result = {}
    with tf.Session() as sess:
        #   Feed the image_data as input to the graph.
        #   predictions  will contain a two-dimensional array, where one
        #   dimension represents the input image count, and the other has
        #   predictions per class
        softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
        predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

        # Sort to show labels in order of confidence
        top_k = predictions.argsort()[-num_top_predictions:][::-1]
        for node_id in top_k:
            human_string = labels[node_id]
            score = predictions[node_id]
            print('{} (score = {:.5f})'.format(human_string, score))
            result[human_string] = score

        return result


def label_image(image_addr, labels_addr='model/output_labels.txt',
                graph_addr='model/output_graph.pb'):
    # load image
    image_data = load_image(image_addr)
    # load labels
    labels = load_labels(labels_addr)
    # load graph, which is stored in the default session
    load_graph(graph_addr)
    return run_graph(image_data, labels)


def fangong_statistic(image_dir):
    images = glob.glob('{}/{}'.format(image_dir, '*'))
    TN = 0  # 将负类预测为负类数
    FP = 0  # 将负类预测为正类数
    for image in images:
        print("Fangong image addr: {}".format(image))
        if label_image(image)['fangong'] > F_THRESHOLD:
            TN += 1
        else:
            FP += 1
    return TN, FP, len(images)


def normal_statistic(image_dir):
    images = glob.glob('{}/{}'.format(image_dir, '*'))
    n_total_image = 0
    TP = 0  # 将正类预测为正类数
    FN = 0  # 将正类预测为负类数
    for image in images:
        print("Normal image addr: {}".format(image))
        if label_image(image)['normal'] > F_THRESHOLD:
            TP += 1
        else:
            FN += 1
    return TP, FN, len(images)


def main():
    # 计算反共黑客测试集的准确率
    # TN, FP, f_total = fangong_statistic("f_data")
    # print('Total fangong image: {}'.format(f_total))
    # print('{} precision = {:.5f}'.format('反共黑客:', TN / f_total))

    # 计算正常图片测试集的准确率
    TP, FN, n_total = normal_statistic("n_data")
    print('Total normal image: {}'.format(n_total))
    print('{} precision = {:.5f}'.format('正常图片:', TP / n_total))

    # 计算整体准确率和召回率
    # f_p = TN / (TN + FN)
    # f_r = TN / (TN + FP)
    # print('The final accuracy: {:.5f}'.format((TN + TP) / (f_total + n_total)))
    # print('The fangong precision: {:.5f}'.format(f_p))
    # print('The fangong recall: {:.5f}'.format(f_r))
    # print('The final F1 score: {:.5f}'.format(2 * f_p * f_r / (f_p + f_r)))


if __name__ == '__main__':
    main()
