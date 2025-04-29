import os
import sys
import numpy as np

def co_occurrence_ratio(seq_i, seq_j):
    if type(seq_j) is not list:
        seq_j = [seq_j]
    if type(seq_j) is not list:
        seq_j = [seq_j]
    if seq_i is None or seq_j is None: return 0
    if len(seq_i) == 0 or len(seq_j) == 0: return 0
    intersection = set(seq_i) & set(seq_j)
    union = set(seq_i) | set(seq_j)
    ratio = len(intersection) / len(union) 
    return ratio

def get_input_seq(seq):
    seq = seq.split('<|history|>')[1].split('<|endofhistory|>')[0].split(' ')
    seq = list(filter(lambda x: x != '', seq))
    return seq

def get_output_seq(seq):
    seq = seq.split('<|pre|>')[1].split('<|endofpre|>')[0].split(' ')
    seq = list(filter(lambda x: x != '', seq))
    seq = list(filter(lambda x: 'time' not in x, seq))
    return seq

def get_inout_list(data, gt):
    in_list = []
    out_list = []
    for i in range(len(data)):
        in_list.append(get_input_seq(data[i]))
        out_list.append(get_output_seq(gt[i]))
    return in_list, out_list

def occurrence_matrix(target, source):
    scores_matrix = np.zeros((len(target), len(source)))
    for i in range(len(target)):
        for j in range(len(source)):
            scores_matrix[i, j] = co_occurrence_ratio(target[i], source[j])
    return scores_matrix

def save_train_annotation(
    scores_matrix,
    scores_matrix_in,
    save_file, 
    save_file_score,
    threshold=0.8,
    neg_num=5
    ):
    cnt = 0
    with open(save_file, 'w') as f, open(save_file_score, 'w') as g:
        for i in range(scores_matrix.shape[0]):
            pos_indices = np.where(scores_matrix[i] > threshold)[0].tolist()
            if len(pos_indices) > 0:
                sorted_indices_in = np.argsort(-scores_matrix_in[i])
                neg_indices_for_instance = []
                count = 0
                for idx in sorted_indices_in:
                    if idx not in pos_indices and scores_matrix[i, idx] > 0:
                        neg_indices_for_instance.append(idx)
                        count += 1
                    if count == neg_num:
                        break
                if len(neg_indices_for_instance) < neg_num:
                    for idx in sorted_indices_in:
                        if idx not in pos_indices and scores_matrix[i, idx] == 0:
                            neg_indices_for_instance.append(idx)
                            count += 1
                        if count == neg_num:
                            break
                # Save the results to files
                if 'dialog' in dataset:
                    pos_indices = pos_indices[:4]
                for pos_ind in pos_indices:
                    #try:
                    #    neg_i = neg_indices_for_instance[i]
                    #except:
                    neg_i = np.random.choice(neg_indices_for_instance)
                    f.write(f"{i} {pos_ind} {neg_i}\n")
                    g.write(f"{i} {scores_matrix[i, pos_ind]} {scores_matrix[i, neg_i]}\n")
                    cnt += 1

    print("Number of original instances:", scores_matrix.shape[0])
    print('Number of positive samples:', cnt)


def save_index_score(score_matrix, save_index_file, save_score_file):
    indices = np.argsort(-score_matrix, axis=1)
    with open(save_index_file, 'w') as f, open(save_score_file, 'w') as g:
        for i in range(score_matrix.shape[0]):
            f.write(' '.join([str(x) for x in indices[i]]) + '\n')
            g.write(' '.join([str(x) for x in score_matrix[i]]) + '\n')



def save_score_file_train(scores_matrix, save_file_index, save_file_score, topk=10):
    with open(save_file_index, 'w') as f_index, open(save_file_score, 'w') as f_score:
        for i in range(scores_matrix.shape[0]):
            score = scores_matrix[i,:]
            topk_indices = np.argsort(-score)[:topk]
            f_index.write( ' '.join(map(str, topk_indices)) + '\n')
            f_score.write(' '.join(map(str, score[topk_indices])) + '\n')





if __name__ == '__main__':

    dataset = sys.argv[1]
    timestamp = sys.argv[2]
    threshold = float(sys.argv[3])
    if threshold == None:
        threshold = 0.8 # default threshold

    save_path = os.path.join('./resources/', dataset, str(timestamp), 'train_retrieval')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # for train retrieval model
    save_file_train_index = os.path.join(save_path, f'train_index.retrieval')
    save_file_train_score = os.path.join(save_path, f'train_score.retrieval')

    save_file_test_index = os.path.join(save_path, f'test_index.retrieval')
    save_file_test_score = os.path.join(save_path, f'test_score.retrieval')
    
    save_file_val_index = os.path.join(save_path, f'val_index.retrieval')
    save_file_val_score = os.path.join(save_path, f'val_score.retrieval')

    # ground truth demonstrations for generator training
    save_path_gen = os.path.join('./resources/train_generator', dataset, str(timestamp), "train_gt_topk")
    if not os.path.exists(save_path_gen):
        os.makedirs(save_path_gen)
    save_topk_index_train = os.path.join(save_path_gen, f'train_index.gen')
    save_topk_score_train = os.path.join(save_path_gen, f'train_score.gen')



    # source data
    train_data_path = os.path.join('resources', dataset, timestamp, 'train.link_prediction')
    test_data_path = os.path.join('resources', dataset, timestamp, 'test.link_prediction')
    test_gt_path = os.path.join('resources', dataset, timestamp, 'test_gt.link_prediction')
    val_data_path = os.path.join('resources', dataset, timestamp, 'val.link_prediction')
    val_gt_path = os.path.join('resources', dataset, timestamp, 'val_gt.link_prediction')

    with open(train_data_path, 'r') as f:
        train_data = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    with open(test_data_path, 'r') as f:
        test_data = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    with open(test_gt_path, 'r') as f:
        test_gt = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    with open(val_data_path, 'r') as f:
        val_data = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    with open(val_gt_path, 'r') as f:
        val_gt = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    
    num_nodes = len(train_data) 
    train_ego_in_seq, train_ego_out_seq = get_inout_list(train_data, train_data)
    test_ego_in_seq, test_ego_out_seq = get_inout_list(test_data, test_gt)
    val_ego_in_seq, val_ego_out_seq = get_inout_list(val_data, val_gt)

    scores_matrix_train_out = occurrence_matrix(train_ego_out_seq, train_ego_out_seq)
    scores_matrix_train_in = occurrence_matrix(train_ego_in_seq, train_ego_in_seq)
    scores_matrix_test = occurrence_matrix(test_ego_out_seq, train_ego_out_seq)
    scores_matrix_val = occurrence_matrix(val_ego_out_seq, train_ego_out_seq)

    # set the diagonal to 0 for scores_matrix_train_out and scores_matrix_train_in
    np.fill_diagonal(scores_matrix_train_out, 0)
    np.fill_diagonal(scores_matrix_train_in, 0)

    scores_matrices = {
        'train_out': scores_matrix_train_out,
        'train_in': scores_matrix_train_in,
        'test': scores_matrix_test,
        'val': scores_matrix_val
    }
    
    # save the files for retriever
    save_train_annotation(scores_matrix_train_out,
                    scores_matrix_train_in, 
                    save_file_train_index, 
                    save_file_train_score,
                    threshold=threshold,
                    neg_num=5)

    save_index_score(scores_matrix_test,
                    save_file_test_index,
                    save_file_test_score)   
    save_index_score(scores_matrix_val,
                    save_file_val_index,
                    save_file_val_score)
    
    # save the train files for generator
    save_score_file_train(scores_matrix_train_out, save_topk_index_train, save_topk_score_train, topk=10)
    
    print("Done!")
    

