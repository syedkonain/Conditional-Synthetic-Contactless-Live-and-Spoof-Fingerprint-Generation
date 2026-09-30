import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TestOptions().parse()  # Get test options
    # Hard-code some parameters for the test
    opt.num_threads = 0   # Test code only supports num_threads = 0
    opt.batch_size = 1    # Test code only supports batch_size = 1
    opt.serial_batches = True  # Disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # No flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # No visdom display; the test code saves the results to an HTML file.
    dataset = create_dataset(opt)  # Create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # Create a model given opt.model and other options
    model.setup(opt)               # Regular setup: load and print networks; create schedulers

    # Initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # Create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # Define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # Test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # Only apply our model to opt.num_test images.
            break
        model.set_input(data)  # Unpack data from data loader
        model.test()           # Run inference
        visuals = model.get_current_visuals()  # Get image results
        img_path = model.get_image_paths()     # Get image paths
        if i % 5 == 0:  # Save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)
    webpage.save()  # Save the HTML

