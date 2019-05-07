# Import the basic packages
import mxnet as mx
from mxnet import nd, init, gluon, autograd, image
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import d2l
CTX = d2l.try_gpu()
import time
import matplotlib.pyplot as plt
import os

# I want an overarching method that trains a VAE against a discriminator
# with the following features:
#
# 1. Tuning training hyperparameters: number of epochs
# 2. Train the first a few epochs with only PBP loss, then train the
#    next a few epochs with PBP loss reduced and Disc loss added
# 3. Record PBP loss, Disc loss, and epoch training time per epoch
#    and write them to a CSV file
# 4. Save VAE model parameters to specified path

def train_VAE_GAN(vae_net,
                  disc_net,
                  train_features,
                  test_features,
                  test_results_dir,
                  vae_parameters_path = None,
                  batch_size = 64,
                  init_lr = 0.001,
                  pbp_weight = 1,
                  disc_loss_mul = 10,
                  n_epochs = 200,
                  n_solo_epochs = 0,
                  max_disc_loss = 999,
                  variable_pbp_weight = 'constant',
                  pbp_weight_decay = 1,
                  CTX = d2l.try_gpu()):
    
    # VAE_net is a VAE network (most likely a ConvVAE with 512 latent variables
    # 32 base channels, 3 * 64 * 64 output shape
    
    # disc_net is a discriminator network (most likely a ResNet)
    # whose output of (batch_size, 1)
    
    # test_results_dir is the directory (must end with a slash /) that contains 
    # the validation images after all epochs were run
    
    # vae_parameters_dir is the path (so directory + filename) that the trained
    # VAE's model parameters will be saved to.
    
    # n_solo_epochs indicate the number of epochs that the VAE will train using
    # no discriminator; n_solo_epochs must be smaller than n_epochs, and 
    # the number of epochs trained with discriminator is
    # n_epochs - n_solo_epochs
    
    # max_disc_loss is the maximum loss beyond which the discriminator's loss
    # will not be used in updating VAE in the generator cycle
    
    # If variable_pbp_weight is False/None, then the pbp weight will remain
    # constant for all epochs (except when training solo, in which case
    # the pbp weight is adjusted to 1, but will revert back to the specified value
    # after solo epochs are done.
    #
    # If variable_pbp_weight is 'decay', then for every 25 combo epochs the
    # pbp_weight will decrease by constant factor.
    
    #############################################################################
    ## MODEL INITIALIZATION AND TRAINER
    #############################################################################
    # 
    # Initialize the VAE network and get its trainer
    print('[STATE]: Initializing model parameters and constructing Gluon trainers')
    # Set the pbp weight to the desired value
    vae_net.pbp_weight = pbp_weight
    vae_net.collect_params().initialize(mx.init.Xavier(), 
                                        force_reinit=True,
                                        ctx=CTX)
    vae_trainer = gluon.Trainer(vae_net.collect_params(),
                                'adam',
                                {'learning_rate': init_lr})
    # Initialize the Disc network nd get its trainer
    disc_net.collect_params().initialize(mx.init.Xavier(),
                                         force_reinit=True,
                                         ctx=CTX)
    disc_trainer = gluon.Trainer(disc_net.collect_params(),
                                 'adam',
                                 {'learning_rate': init_lr})
    
    #############################################################################
    ## Output file writer initialization
    #############################################################################
    #
    # Open a file to write to for training statistics; the training statistics csv
    # will be written to the results directory
    csv_writer = None
    try:
        csv_writer = open(test_results_dir + 'training_statistics.csv', 'w')
        print('[STATE]: Writing training statistics to ' + test_results_dir + 'training_statistics.csv')
    except:
        print('[ERROR]: test results directory not valid, writing training statistics to main directory')
        csv_writer = open('./training_statistics.csv', 'w')
    # CSV file needs to open with a header that is the column names
    csv_writer.write('epoch,vae_loss,disc_loss,time_consumed\n')
        
    # Open a file to write README.md for displaying validation images; the README
    # file will be written to the results directory
    readme_writer = None
    try:
        readme_writer = open(test_results_dir + 'README.md', 'w')
        print('[STATE]: Writing README report to ' + test_results_dir + 'README.md')
    except:
        print('[ERROR]: test results directory not valid, writing readme to main directory')
        csv_writer = open('./README.md', 'w')
    # Write a few lines on README to indicate the hyper parameters
    readme_writer.write('n_latent:{} \n\n'.format(vae_net.n_latent))
    readme_writer.write('n_base_channels:{} \n\n'.format(vae_net.n_base_channels))
    if variable_pbp_weight == 'constant':
        readme_writer.write('pixel-by-pixel loss weight:{} \n\n'.format(vae_net.pbp_weight))
    elif variable_pbp_weight == 'decay':
        readme_writer.write('pixel-by-pixel loss weight initially {} and decay by {} every 25 combo epochs \n\n'.format(vae_net.pbp_weight, pbp_weight_decay))
    readme_writer.write('n_solo_epochs:{} \n\n'.format(n_solo_epochs))
    readme_writer.write('n_combo_epochs:{} \n\n'.format(n_epochs - n_solo_epochs))
    readme_writer.write('max_disc_loss :{} \n\n'.format(max_disc_loss))
        
    #############################################################################
    ## Data iterator 
    #############################################################################
    #
    # Load training features into an iterator
    train_iter = gdata.DataLoader(train_features,
                                  batch_size,
                                  shuffle=True,
                                  last_batch='keep')
    sample_size = train_features.shape[0]
    print('[STATE]: {} training samples loaded into iterator'.format(sample_size))
    
    #############################################################################
    ## Training parameters
    #############################################################################
    #
    # Figure out the number of epochs trained with VAE and Discriminator together
    n_combo_epochs = n_epochs - n_solo_epochs
    print('[STATE]: {} solo epochs and {} combo epochs are to be trained'.format(n_solo_epochs,
                                                                                 n_combo_epochs))
    
    #############################################################################
    ## Training
    #############################################################################
    #
    # Print a message, then start training
    print('[STATE]: Training started')
    
    # First train the solo rounds; when training the solo rounds, PBP weight
    # is 1; however it needs to be changed back to the specified constant
    # when ConvVAE is initialized, or a variable PBP weight, so I will keep
    # a copy of the specified PBP weight to be used later
    specified_pbp_weight = vae_net.pbp_weight
    vae_net.pbp_weight = 1
    for epoch in range(n_solo_epochs):
        
        # Initialize a list that records the average VAE loss per batch
        batch_losses = []
        epoch_start_time = time.time()
        
        # Iterate through the epochs
        for batch_features in train_iter:
            # Load the batch into the appropriate context
            batch_features = batch_features.as_in_context(CTX)
            
            # Compute loss, gradient, and update paramters using trainer
            with autograd.record():
                loss = vae_net(batch_features)
                loss.backward()
            vae_trainer.step(batch_features.shape[0])
            # Compute the mean loss among the batch, append it onto the batch
            # losses list
            batch_losses.append(nd.mean(loss).asscalar())
        
        # Compute the training loss per epoch by a mean of batch losses
        epoch_train_loss = np.mean(batch_losses)
        epoch_stop_time = time.time()
        time_consumed = epoch_stop_time - epoch_start_time
        
        # Generate the epoch report and write it to the README file
        # and print it
        epoch_report_str = 'Epoch{}, Training loss {:.10f}, Time used {:.2f}'.format(epoch,
                                                                                     epoch_train_loss,
                                                                                     time_consumed)
        readme_writer.write(epoch_report_str + '\n\n')
        print('[STATE]: ' + epoch_report_str)
        
    # Now that all solo rounds are over, revert the PBP weight of the vae back to the original
    # specified value
    vae_net.pbp_weight = specified_pbp_weight
        
    # Now train the combo rounds; we will use BinarySigmoidCrossEntropyLoss()
    # for discriminator loss
    disc_loss_func = gloss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    
    # Define an integer that is 0 if discriminator loss in an epoch is larger than
    # the specified max_disc_loss (discriminator is bad, don't follow it)
    # Before any training, set it to 1
    use_disc_loss = 1
    for epoch in range(n_solo_epochs, n_epochs):
        start_time = time.time()
        
        # Initialize the lists that records the average loss within each batch
        vae_batch_losses = []
        disc_batch_losses = []
        
        # Iterate through the batches
        for batch_features in train_iter:
            # Load the batch into the appropriate context
            batch_features = batch_features.as_in_context(CTX)
            # Record the batch_size because it may not be the specified batch size
            act_batch_size = batch_features.shape[0]
            
            # Generate some 1s and 0s for distinguishing genuine images from
            # generated images
            genuine_labels = nd.ones((act_batch_size,), ctx=CTX)
            generated_labels = nd.zeros((act_batch_size,), ctx=CTX)
            
            ############################################################################
            # UPDATE THE DISCRIMINATOR NETWORK
            ############################################################################
            with autograd.record():
                
                # Train with genuine images: make predictions on genuine images
                genuine_logit_preds = disc_net(batch_features)
                genuine_loss = disc_loss_func(genuine_logit_preds,
                                                  genuine_labels)
                
                # Train with generated images: make predictions on generated images
                generated_features = vae_net.generate(batch_features)
                generated_logit_preds = disc_net(generated_features)
                generated_loss = disc_loss_func(generated_logit_preds, 
                                                generated_labels)
                
                # Total loss is loss with genuine and with generated images
                disc_loss = genuine_loss + generated_loss
                disc_loss.backward()
                disc_batch_losses.append(nd.mean(disc_loss).asscalar())
                
            # Update the parameters in the convolutional discriminator
            disc_trainer.step(act_batch_size)
            
            ############################################################################
            # UPDATE THE VAE NETWORK
            ############################################################################
            with autograd.record():
                
                # Compute the discriminator loss by letting the discriminator network
                # make predictions on the generated images
                generated_features = vae_net.generate(batch_features)
                generated_logit_preds = disc_net(generated_features)
                batch_disc_loss = disc_loss_func(generated_logit_preds, genuine_labels)
                
                # Sum up the VAE loss and the discriminator loss (with multiplier of 10)
                # Then multiply batch_disc_loss by an integer
                # that is 1 if 
                gen_loss = vae_net(batch_features) + batch_disc_loss * disc_loss_mul * use_disc_loss
                gen_loss.backward()
                
                # Record the VAE batch loss's average
                vae_batch_losses.append(nd.mean(gen_loss).asscalar())
                
            # Update the parameters in the VAE network
            vae_trainer.step(act_batch_size)
            
        ############################################################################
        # NEAR THE END OF THIS EPOCH
        ############################################################################
        
        # Compute some summarical metrics of this epoch
        stop_time = time.time()
        time_consumed = stop_time - start_time
        epoch_disc_train_loss = np.mean(disc_batch_losses)
        epoch_vae_train_loss = np.mean(vae_batch_losses)
        
        # If variable_pbp_weight is set to decay, then decay the pbp weight
        if variable_pbp_weight == 'decay':
            if (1+epoch) % 25 == 0:
                vae_net.pbp_weight = vae_net.pbp_weight * pbp_weight_decay
                print('VAE PBP weight adjusted to {:.10f}'.format(vae_net.pbp_weight))
            
        # Check if discriminator is good enough at the end of this epoch
        # if good enough, keep use_disc_loss at 1
        if epoch_disc_train_loss <= max_disc_loss:
            use_disc_loss = 1
        else:
            # Note that even if use_disc_loss is set to 0
            # discriminator will still be trained in the next epoch,
            # just its loss not used in the VAE update cycle
            use_disc_loss = 0
        
        # Generate the README line and the csv line, and write them
        epoch_README_report = 'Epoch{}, VAE Training loss {:.5f}, ResNet Training loss {:.10f}, Time used {:.2f}'
        epoch_README_report = epoch_README_report.format(epoch,
                                                         epoch_vae_train_loss,
                                                         epoch_disc_train_loss,
                                                         time_consumed)
        epoch_CSV_report = '{},{:.10f},{:.10f},{:.2f}'.format(epoch,
                                                             epoch_vae_train_loss,
                                                             epoch_disc_train_loss,
                                                             time_consumed)
        readme_writer.write(epoch_README_report + '\n\n')
        csv_writer.write(epoch_CSV_report + '\n')
        print('[STATE]: ' + epoch_README_report)
        
    ############################################################################
    # END OF TRAINING, now onto the validation process
    ############################################################################
    # Close the CSV writer because there is nothing left to write
    csv_writer.close()
    
    # Save model parameters; if vae_parameters_path is not valid, do not save it
    try:
        vae_net.save_parameters(vae_parameters_path)
    except:
        print('[ERROR]: VAE parameters path is not valid; parameters will be saved to main directory')
        vae_net.save_parameters('./recent_model.params')
    
    # Define the number of validation images to generate
    # then use the vae_net to generate them
    n_validations = 10
    img_arrays = vae_net.generate(test_features[0:n_validations].as_in_context(CTX)).asnumpy()
    
    for i in range(n_validations):
        # Write a line in the README report the displaying the generated images
        readme_writer.write('!['+str(i)+'](./'+str(i)+'.png)')
        
        # Reshape the output from (n_channels, width, height) to (width, height, n_channels)
        # Note that the vae_net instance already has such information regarding
        # the training images
        img_array = img_arrays[i].reshape((vae_net.out_width,
                                        vae_net.out_height,
                                        vae_net.n_channels))
        
        # Show the plot, save it. If test_results_dir is not valid,
        # save it to main directory
        plt.imshow(img_array)
        try:
            plt.savefig(test_results_dir + str(i) + '.png')
            print('[STATE]: ' + test_results_dir + str(i) + '.png' + ' saved')
          
        except:
            print('[ERROR]: test results directory not valid, saving images to main directory')
            plt.savefig('./' + str(i) + '.png')
        plt.close()
        
    # Close the README writer
    readme_writer.close()
        




























    
    
                  