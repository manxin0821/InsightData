import argparse
import predict
import sentiment
import similarity
from pathlib import Path
import logging

class ArgumentDefaultsHelpFormatter_RawDescription(argparse.ArgumentDefaultsHelpFormatter):
    def _fill_text(self, text, width, indent):
        return ''.join(indent + line for line in text.splitlines(keepends=True))
    
def parse(args=None):
    parser = argparse.ArgumentParser(description=" Test single sentence or check the full validation results. Example: python scripts/run_proj.py --option 'validation' ")
    parser.add_argument('options', type=str, nargs = '?', help = 'training, validation or testing')
    parser.add_argument('--data_directory', default=Path('data'), type=Path, nargs='?', help='Data directory (default: ./data/)')
    parser.add_argument('--model_directory', default=Path('models'), type=Path, nargs='?', help='Model directory (default: ./model/)')    
    parser.add_argument('--output_directory', default=Path('output'), type=Path, nargs='?', help='Output directory (default: ./output/)')
    parser.add_argument('--input_sentence', type=str, nargs='?', help='Input sentence')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Verbose (add output);  can be specificed multiple times to increase verbosity')

    return parser.parse_args(args) 
    
def main():   
    args = parse()
    if 0 == args.verbose:
        ll = logging.ERROR
    elif 1 == args.verbose:
        ll = logging.INFO
    else:
        ll = logging.DEBUG

    
    if args.options == 'training':
        sentiment.get_logger(f'{args.options}.log', ll, args.output_directory/'logs/')
        sentiment.run(args.data_directory, args.model_directory)
        similarity.get_logger(f'{args.options}.log', ll, args.output_directory/'logs/')
        similarity.run(args.data_directory, args.model_directory)
        
    elif args.options == 'validation':
        predict.get_logger(f'{args.options}.log', ll, args.output_directory/'logs/')
        predict.run_validation(args.data_directory, args.model_directory, args.output_directory)
    else:
        predict.get_logger(f'{args.options}.log', ll, args.output_directory/'logs/')
        predict.run_single(args.data_directory, args.model_directory, args.input_sentence)
     
if __name__ == '__main__':
    main()