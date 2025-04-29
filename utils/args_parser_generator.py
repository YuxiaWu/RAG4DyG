
import argparse
import os   

class ArgsParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--lrdecay", type=int, default=0, help="Whether to use lr decay.")
        parser.add_argument("--run_seed", action="store_true", help="Whether to run on different seed.")
        parser.add_argument('--n_gpu', default=1)
        parser.add_argument("--timestamp", default=None, type=str, required=True, help="The timestamp to deal with.")          
        parser.add_argument("--dataset", default=None, type=str, required=True, help="The datasest name.")        
        parser.add_argument("--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file).")
        parser.add_argument("--output_dir",type=str,required=True,help="The output directory where the model predictions and checkpoints will be written.",)
        parser.add_argument("--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",)
        parser.add_argument(
            "--eval_data_file",
            default=None,
            type=str,
            help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
        )

        parser.add_argument(
            "--eval_data_gt_file",
            default=None,
            type=str,
            help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
        )
        parser.add_argument(
            "--test_data_file",
            default=None,
            type=str,
            help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
        )

        parser.add_argument(
            "--test_data_gt_file",
            default=None,
            type=str,
            help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
        )

        parser.add_argument(
            "--train_index_file", 
            default=None,
            type=str,
            help="The input training index file (a text file).",
        )

        parser.add_argument(
            "--train_score_file",
            default=None,
            type=str,
            help="The input training score file (a text file).",
        )

        parser.add_argument(
            "--val_index_file",
            default=None,
            type=str,
            help="The input valuation index file (a text file).",
        )

        parser.add_argument(
            "--val_score_file",
            default=None,
            type=str,
            help="The input valuation score file (a text file).",
        )

        parser.add_argument(
            "--test_index_file",
            default=None,
            type=str,
            help="The input test index file (a text file).",
        )

        parser.add_argument(
            "--test_score_file",
            default=None,
            type=str,
            help="The input test score file (a text file).",
        )

        parser.add_argument(
            "--retrieval_checkpoint",
            default=None,
            type=str,
            help="The model checkpoint for retrieval.",
        )
        parser.add_argument(
            "--simpledyg_checkpoint",
            default=None,
            type=str,
            help="The model checkpoint for SimpleDyG.",
        )
        # for fune-tuning parameters
        parser.add_argument("--n_layer",default=12,type=int,help="number of layers")
        parser.add_argument("--n_head",default=12,type=int,help="number of heads")
        parser.add_argument("--n_embed",default=768,type=int,help="embedding dimension")
        parser.add_argument("--mlp_layers", default=2, type=int, help="number of mlp layers")
        parser.add_argument("--gnn_layers", default=1, type=int, help="number of gnn layers")
        parser.add_argument("--fusion", default='mlp', type=str, help="the fusion strategy after retrieval")
        parser.add_argument("--negative", default=0,type=int,  help="use how many negative samples")
        parser.add_argument("--temperature", default=0.1,type=float, help="temperature for CL loss")
        parser.add_argument("--node_feat_file",default=None,type=str,help="original node feature.",)
        parser.add_argument("--topK", type=int, default=7, help="topK retrieval results")
        parser.add_argument("--m",type=int, default=10, help="number of projected embeddings for MLP fusion")
        parser.add_argument("--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir")
        parser.add_argument("--freeze", action="store_true", help="Whether to freeze the transformer in SimpleDyG")
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
        )

        parser.add_argument(
            "--config_name",
            default=None,
            type=str,
            help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
        )
        parser.add_argument(
            "--tokenizer_name",
            default=None,
            type=str,
            help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
        )
        parser.add_argument(
            "--cache_dir",
            default=None,
            type=str,
            help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
        )
        parser.add_argument(
            "--block_size",
            default=-1,
            type=int,
            help="Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens).",
        )
        parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
        parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
        parser.add_argument(
            "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
        )

        parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
        parser.add_argument(
            "--per_gpu_eval_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation."
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumulate before performing a backward/update pass.",
        )
        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
        parser.add_argument("--weight_decay", default=1e-5, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
        parser.add_argument(
            "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
        )
        parser.add_argument(
            "--max_steps",
            default=-1,
            type=int,
            help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
        )
        parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
        parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
        parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
        parser.add_argument(
            "--save_total_limit",
            type=int,
            default=None,
            help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
        )
        parser.add_argument(
            "--eval_all_checkpoints",
            action="store_true",
            help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
        )
        parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
        parser.add_argument(
            "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
        )
        parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")



        parser.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
        )
        parser.add_argument(
            "--fp16_opt_level",
            type=str,
            default="O1",
            help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
            "See details at https://nvidia.github.io/apex/amp.html",
        )
        parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        parser.add_argument("--patience", type=int, default=5, help="early stop patience")

        self.parser = parser
    def parse(self):
        args = self.parser.parse_args()
        return args