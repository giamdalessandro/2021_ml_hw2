import sys, os.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import json
import torch
from numpy import mean

from small_data import available_datasets, get_dataset, get_pipeline
from small_data.evaluation import balanced_accuracy


def load_model(params_file, weights_file, data):

    with open(params_file) as f:
        params = json.load(f)
    
    pipeline = get_pipeline(params['method'], **params['hparams'])
    _, transform = pipeline.get_data_transforms(data)
    
    model = pipeline.create_model(params['architecture'], data.num_classes, data.num_input_channels)
    model.load_state_dict(torch.load(weights_file))
    model = model.cuda()
    
    return model, transform


if __name__ == '__main__':

    # Parse Arguments
    parser = argparse.ArgumentParser(
        description='Compute the average balanced accuracy from N repetitions of the same method on a given dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dataset', type=str, choices=available_datasets(),
                        help='Name of the dataset.')
    parser.add_argument('paramsfile', type=str,
                        help='Path to a JSON file containing the keys "method", "architecture", and "hparams" defining the pipeline of the reference model. ' \
                             'The --history file from train.py would be a suitable candidate. ' \
                             'Note that all the models are assumed to have the same configuration.')
    parser.add_argument('--methodpattern', type=str, default=None,
                        help='Pattern of the file name to search for the balanced accuracy evaluation. ' \
                              'Models are expected to be saved as {NAME}_REPID.pth. ' \
                              'Therefore, the correct pattern to provide in this case is NAME')
    parser.add_argument('--modelsfolder', type=str, default=None,
                    help='Path to a folder containing the models weights. ' \
                         'Note that the names of the .pth file should match the ones of the method (e.g. {NAME}_REPID.pth')
    parser.add_argument('--modelfile', type=str, default=None,
                        help='Path to a single model file to be evaluated. '\
                             'If this is set, --methodpattern and --modelsfolder have no effect.')
    group = parser.add_argument_group('Data')
    group.add_argument('--data-root', type=str, default=None,
                       help='Dataset root directory containing the split files. '\
                            'Defaults to "./datasets/<dataset>".')
    group.add_argument('--img-dir', type=str, default=None,
                       help='The directory containg the images, if different from the default.')
    group.add_argument('--test-split', type=str, default='test',
                       help='File specifying the subset of the data to be used for evaluation.')
    group.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for the evaluation.')
    args = parser.parse_args()
    
    if args.data_root is None:
        args.data_root = os.path.join('datasets', 'cifair' if args.dataset.startswith('cifair') else args.dataset)

    # Load dataset
    data_kwargs= {}
    if args.img_dir is not None:
        data_kwargs['img_dir'] = args.img_dir
    test_data = get_dataset(args.dataset, args.data_root, args.test_split, **data_kwargs)
    
    if args.modelfile is not None:
        
        # Evaluate single model
        model, transform = load_model(args.paramsfile, args.modelfile, test_data)
        test_data.transform = transform
        bacc = balanced_accuracy(model, test_data, args.batch_size)
        print(bacc)
    
    elif (args.modelsfolder is not None) and (args.methodpattern is not None):

        # Evaluate multiple models
        models_files = os.listdir(args.modelsfolder)
        accs = []
        
        for mf in models_files:

            try:
                repid = int(mf.rsplit(args.methodpattern + '_', 1)[-1].split('.pth')[0])
            except:
                continue
        
            # Load model
            model, transform = load_model(args.paramsfile, args.modelsfolder + mf, test_data)
            test_data.transform = transform
            
            # Evaluate balanced accuracy
            bacc = balanced_accuracy(model, test_data, args.batch_size)
            
            print("Model ID: {} -- Balanced Accuracy {}".format(repid, bacc))
            
            accs.append(bacc)
        
        avg_bacc = mean(accs)
        print('Average balanced accuracy: {:.2%}'.format(avg_bacc))
    
    else:
        
        print('Either --modelfile or --modelsfolder and --methodpattern need to be specified.')
        exit(-1)
