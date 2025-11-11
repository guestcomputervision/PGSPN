
# Hyperparmeter Setting
depth_range = dict(min_depth_eval=0.001)

image_size = dict(input_height=576,
                  input_width=960
                  )

train_parm = dict(num_threads=4,
                  batch_size=3,
                  num_epochs=12,                  
                  stage1_checkpoint_path='checkpoints/transmission_estimator.pth',
                  stage2_checkpoint_path=None,
                  is_retrict=False,
                  retrain=True,
                  learning_rate=3e-4,  
                  )

test_parm = dict(stage1_test_checkpoint_path='checkpoints/transmission_estimator.pth',
                 stage2_test_checkpoint_path='checkpoints/depth_completion.pth',
                 is_save_gt_image=False,
                 is_save_input_image=False,
                 )


freq = dict(save_freq=1000,
            log_freq=250,
            eval_freq=2500
            )


etc = dict(is_checkpoint_save=True,
           do_mixed_precison=False,
           do_online_eval=True,
           do_use_logger='Tensorboard'
           )


# Log & Save Setting
log_save_cfg = dict(save_root='save',
                    log_directory='log',
                    eval_log_directory='eval',
                    model_save_directory ='checkpoints',
                    )

# Basic Setting
basic_cfg = dict(
               model_stage1_cfg = dict(type='Build_Structure',
                                        structure_cfg = dict(type='Learning_04_04_step1',
                                                             enc='res34',        # res18, res34, res68
                                                             suffle_up=False,
                                                             sto=True,
                                                             iteration=5
                                                             ),
                                        ),
               model_stage2_cfg = dict(type='Build_Structure',
                                  structure_cfg = dict(type='OursLRRU_1',
                                                       use_depth_similarity=True,
                                                       similarity_temperature=1.0, 
                                                       normalize_mode='threshold', 
                                                       relative_threshold=0.3, 
                                                       decay_rate=1.2
                                                      )
                                  ),
               loss_build_cfg = dict(type='Builder_Loss',
                                       loss_build_list = [                     
                                                          dict(type='Build_DepthCompletion_Loss', 
                                                               depth_min_eval=depth_range['min_depth_eval'],
                                                               total_loss_lamda = 10.0,
                                                               loss_cfg_list=[   
                                                                              dict(type='Iter_Depth_Estimation_L1_loss', 
                                                                                   tag='dc_iter_l1_loss',
                                                                                   lambda_l1=20.0,
                                                                                   gamma=0.7
                                                                                   )
                                                                             ],
                                                               ),
                                                          ]),
                 )                 
