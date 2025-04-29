import math
import torch
import os
import json
import pandas as pd
from dataloader.generator import load_and_cache_examples

class Evaluation:
	def jaccard(self, pred, label):
		pred = set(pred)
		label = set(label)
		return len(pred & label) / len(pred | label)

	def ndcg_k(self,sorted_indices, ground_truth, k):
		dcg, pdcg = 0,0
		for i, item in enumerate(sorted_indices[:k]):
			if item in ground_truth:
				dcg += 1 / math.log(i + 2)
		for i in range(min(len(ground_truth), k)):
			pdcg += 1 / math.log(i + 2)
		return dcg / pdcg

	def map_k(self,sort, y, k):
		sum_precs = 0
		ranked_list = sort[:k]
		hists = 0
		for n in range(len(ranked_list)):
			if ranked_list[n] in y:
				hists += 1
				sum_precs += hists / (n + 1)
		return sum_precs

	def recall_k(self,sort, y, k):
		recall_correct = 0
		for y_i in y:
			if y_i in sort[:k]:
				recall_correct += 1
		return recall_correct/len(y)

	def precision_k(self, sort, y, k):
		precision_correct = 0
		for y_i in y:
			if y_i in sort[:k]:
				precision_correct += 1
		return precision_correct/k        



def get_eval_metrics_generator(args, epoch, model, tokenizer, step, mode = "val", is_rag = False, is_best=False):

    if args.fusion=="mlp":
        from utils.model import fusion_mlp as fusion_v

    if args.fusion=="graphpooling":
        from utils.model import fusion_graphpooling as fusion_v

    spl_tokens = tokenizer.additional_special_tokens+[tokenizer.bos_token,tokenizer.eos_token,tokenizer.pad_token]
    #print('spl_tokens: ', spl_tokens) 
    if mode=='val':
        file_path = args.eval_data_file
        file_path_gt = args.eval_data_gt_file
        file_path_score = args.val_score_file
        file_path_index = args.val_index_file
    elif mode=='test':
        file_path = args.test_data_file
        file_path_gt = args.test_data_gt_file
        file_path_score = args.test_score_file
        file_path_index = args.test_index_file

    with open(file_path, encoding="utf-8") as f:
        data = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    with open(file_path_gt, encoding="utf-8") as f:
        data_gt = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    with open(file_path_score, encoding="utf-8") as f:
        data_score = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
    
    with open(file_path_index, encoding="utf-8") as f:
        data_index = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    # make sure the length of data and data_gt are the same
    assert len(data) == len(data_gt)  

    # read the vocab of this timestamp, when evluating, omit the ids not in the vocab
    timestamp = args.timestamp
    dataset = args.dataset

    vocab_file = os.path.join('./vocabs', dataset, timestamp, 'vocab.json')
    vocab = json.load(open(vocab_file, 'r'))

    if args.do_train:
        indicator = 'do_train'
        root_path = 'rag_results/train_mode'
    else:
        indicator = 'do_val'

        root_path = 'rag_results/val_mode'

    if args.run_seed:
        output_generation_path = os.path.join(root_path, dataset, timestamp, args.run_name, "results_seed", mode+'_score')
    else:
        output_generation_path = os.path.join(root_path, dataset, timestamp, args.run_name,  "results", mode+'_score')

    if not os.path.exists(output_generation_path):
        os.makedirs(output_generation_path)

    Eval = Evaluation()   
    model.eval()
    model.to('cuda')

    break_tokens = tokenizer.encode("<|endoftext|>")
    MAX_LEN = model.config.n_ctx
    print('MAX_LEN: ', MAX_LEN)

    generated_dict = {}
    topk = [5]
    metric_terms = ['R', 'NDCG','jaccard']
    top_k_scores = {metric: len(topk)*[0] for metric in metric_terms}

    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

    num_user_test = 0
    with torch.no_grad():
        for i, (input_text, text_gt, index, score) in enumerate(zip(data, data_gt, data_index, data_score)):
            index = list(map(int, index.split()))
            score = list(map(float, score.split())) 
            index = torch.tensor(index).to('cuda') # this is the index of the similar cases after ranking
            score = torch.tensor(score).to('cuda') # this the the similar scores of all the candidate cases without ranking

            is_rag = True

            generated_dict[i] = {}   
            user_id = input_text.split()[2] # is the egoid
            target_list = text_gt.split()[1:-2]
            target_list = [token for token in target_list if token != user_id]
            target_list = [token for token in target_list if token in vocab]

            if len(target_list) == 0:
                print('text_gt: ', text_gt)
                continue
            indexed_tokens = tokenizer.encode(input_text)
            num_user_test+=1 
            if len(indexed_tokens) > MAX_LEN:
                print('len_input: ', len(indexed_tokens))
                indexed_tokens = indexed_tokens[-1000:]
            tokens_tensor = torch.tensor([indexed_tokens])
            # If you have a GPU, put everything on cuda
            tokens_tensor = tokens_tensor.to('cuda')
            predicted_index = []
            len_input = len(indexed_tokens)
            gen_len = 0
            while predicted_index not in break_tokens:
                # Inference: new and old model have different outputs
                if is_rag:
                    predictions = fusion_v(args, model, tokenizer, train_dataset, tokens_tensor, index, args.m, top_k=args.topK)
                else:
                    outputs,_ = model(tokens_tensor)
                    predictions = outputs[0]
                predicted_index = torch.argmax(predictions[0, -1, :]).item() 
                indexed_tokens += [predicted_index]
                
                predicted_text = tokenizer.decode(indexed_tokens)
                
                gen_len += 1
                
                tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')
                if mode=="val":
                    if gen_len>10:
                        break
                else:
                    if len(indexed_tokens) >= MAX_LEN-len(spl_tokens):
                        break            
                if tokenizer.decode(indexed_tokens).endswith('<|endoftext|>'):
                    break

            predicted_text = tokenizer.decode(indexed_tokens)
            predicted_list = predicted_text.split()
            predicted_list = predicted_list[len_input:]
            predicted = [token for token in predicted_list if token != user_id]
            predicted = [token for token in predicted if token not in spl_tokens]
            
            # ndcg
            if 'NDCG' in metric_terms:
                for topi, k in enumerate(topk):
                    try:
                        result = Eval.ndcg_k(predicted, target_list, k)
                        top_k_scores['NDCG'][topi] += result
                    except:
                        print('predicted: ', predicted)
                        print('target_list: ', target_list)
                        top_k_scores['NDCG'][topi] += 0 
            if 'jaccard' in metric_terms:
                result = Eval.jaccard(predicted, target_list)
                for topi, k in enumerate(topk):
                    top_k_scores['jaccard'][topi] += result

            # recall
            if 'R' in metric_terms:
                for topi, k in enumerate(topk):
                    try:
                        result = Eval.recall_k(predicted, target_list, k)
                        top_k_scores['R'][topi] += result
                    except:
                        pass
            generated_dict[i]['user_id'] = user_id
            generated_dict[i]['input'] = input_text
            generated_dict[i]['target_list'] = target_list
            generated_dict[i]['len input_text'] = len(input_text.split())
            generated_dict[i]['predicted_list_ori'] = predicted_list
            generated_dict[i]['predicted'] = predicted
            generated_dict[i]['NDCG@k'] = str( Eval.ndcg_k(predicted, target_list, 1))
            generated_dict[i]['num_user_test'] = str(num_user_test)

        for metric in metric_terms:
            for topi, k in enumerate(topk):
                top_k_scores[metric][topi] = round(top_k_scores[metric][topi] / num_user_test, 4)
        
        result_save_file = os.path.join(output_generation_path, mode+'_results_epoch.csv')

        if is_best:
            with open(result_save_file, 'w') as f:
                f.write('epoch' + ',')
                for para_name in args.para_names:
                    f.write(para_name + ',')
                for k in topk:
                    f.write('R@{},'.format(k))
                for k in topk:
                    f.write('NDCG@{},'.format(k))
                for k in topk:
                    f.write('jaccard@{},'.format(k)) 
                f.write('\n')
                f.write(str(epoch) + ',')
                for para_value in args.para_values:
                    f.write(str(para_value) + ',')
                if "R" in top_k_scores:
                    for ind_k, k in enumerate(topk):
                        f.write(str(top_k_scores['R'][ind_k]) + ',')
                if "NDCG" in top_k_scores:
                    for ind_k, k in enumerate(topk):
                        f.write(str(top_k_scores['NDCG'][ind_k]) + ',')    
                if "jaccard" in top_k_scores:
                    for ind_k, k in enumerate(topk):
                        f.write(str(top_k_scores['jaccard'][ind_k]) + ',')         
                f.write('\n')
                f.flush()
        with open('{}.json'.format(output_generation_path +'/eval_results'), 'wt') as f:
            json.dump(generated_dict, f, indent=4)

        if is_best:
            if args.run_seed:
                save_folder = os.path.join(indicator, mode+'_metrics_seed_gen')
            else:
                save_folder = os.path.join(indicator, mode+'_metrics_ft_gen')
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            result_save_test = os.path.join(save_folder, dataset + '_SimpleDyG.csv')
            test_results = pd.read_csv(result_save_file)

            if os.path.exists(result_save_test):
                test_results.to_csv(result_save_test, mode='a', header=False, index=False)
            else:
                test_results.to_csv(result_save_test, index=False)

    return top_k_scores