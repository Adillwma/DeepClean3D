





- High priority
  - Fix test_epoch / test_epoch2 using undefined `epoch`:
    - Remove or pass `epoch` where used (e.g., pixel telemetry block) or compute telemetry in caller.
  - full_model_export returns None when `checkpoint_save` is True:
    - Ensure function always returns AE file name (add explicit return in both branches).
  - `shuffle_train_data` flagged as breaking (NaNs in ACBMSE / FP16 behaviour):
    - Investigate root cause and add guarded tests; add a reproducible minimal test-case.
  - Inconsistent loss_function_selection checks:
    - File writes check numeric values (e.g., `if loss_function_selection == 0`) but selection is a string — normalize and fix checks.

- Data loader / preprocessing
  - `data_loader_workers = 0# FIX!` — decide default and handle edge cases for Windows (multiprocessing issues).
  - `preprocess_on_gpu` handling — ensure fallback to CPU when CUDA unavailable and avoid double .to(device) moves.
  - `train_loader` / `test_loader` created twice (first with batch_size, then replaced with batch_sampler) — consolidate and remove duplication.
  - `num_of_files_in_path` calculation multiplies by `large_data_bundles_size` — confirm logic for bundle-based datasets and add unit tests.
  - `MTN_Dataset` / StackedDatasetLoader: memory/batching comments — audit memory duplication and document expected memory footprint.

- Precision / device / dtype issues
  - Precision/dtype handling inconsistent when loading pretrained model:
    - `precision = 32 #checkpoint["precision"] BUG` and `dtype = torch.float32   # remove once fixed!!!!` — use consistent validation, convert strings to torch dtypes, and propagate reliably.
  - FP16 / mixed precision NaN handling:
    - Add checks, automatic fallback, or gradient scaling when using 16-bit; document supported hardware.

- Loss, renormalisation, metrics
  - `belief_telemetry` returns tensors; comment suggests returning ints:
    - Cast to Python int (.item()) before returning.
  - `renorm_for_loss_calc` branching duplicates renorm operations in several places — centralise renorm logic and avoid in-place modification of tensors (use .clone()/.detach()).
  - Some custom loss functions marked DEFUNCT / NEEDS CLEANING (ff_loss, loss_penalty, loss_reduction) — remove or refactor.
  - Split loss selection indices vs names inconsistent — normalize interface (use strings or enums).

- Plotting, telemetry and visualisation
  - `plot_epoch_data` & reconstruct_3D:
    - 3D masking plotting sometimes raises errors when masked_im shape mismatches — add robust shape checks and clearer error messages.
  - `plot_pixel_threshold_telemetry` uses variables not passed into functions (e.g., `decoded_data`, `epoch` inside test_epoch) — unify where telemetry is generated.
  - `plot_or_save` semantics mixed between ints and strings; standardize enum/constants and validate inputs.
  - `plot_live` comparatives: trimming of comparative epoch lists is fragile (`epoch_t_list = epoch_t_list[1:]`) — fix source of extra 0 time entries.

- Model export / pretrained handling
  - Robust_model_export2 usage: comments suggest brittle detection and hard-coded AE_file_name:
    - Simplify: ensure returned script and class names are reliable, remove hard-coded fallbacks.
  - Loading pretrained checkpoint via exec():
    - Add security checks, version tagging, and fallbacks if expected classes or fields are missing.
  - full_model_export: duplicate save logic and path handling is confusing — refactor to single save path builder and document outputs.

- Hooks, telemetry buffering and memory
  - hook_flush_interval/hook_max_memory_usage marked FIX / CONNECT:
    - Implement safe flushing, memory tracking, and a clear API for enabling/disabling.
  - register_network_hooks writes into globals `activations`, `weights_data`, `biases_data` — consider encapsulating and limiting memory growth.

- Profiling, compile and platform caveats
  - `compile_model` comment "NOT AVAILIBE ON WINDOWS?!" — detect platform and disable/guard torch.compile accordingly.
  - `run_pytorch_profiler` uses hard-coded trace path '/pytorch_profiler_tracelog' — make configurable and validate write permissions.
  - cProfile results path and snakeviz call rely on snakeviz installation — make optional and check availability.

- Imports and unused code
  - Remove or justify odd imports: `from pickle import FALSE`, `from sympy import use`, `from xml.etree.ElementInclude import include` — these appear unused or incorrect.
  - Consolidate duplicate imports (`torch` imported multiple times) and unused variables.

- Code quality / structure / TODOs
  - Many "CLEAN UP THIS METHOD TO SOMTHING BETTER!!!!!!" comments:
    - Refactor large functions into smaller units (e.g., plotting, metric aggregation, training loop).
  - Centralise settings and CLI override:
    - `override_globals_with_cli()` currently overrides many globals including complex types — validate/parse lists/dicts robustly and support typed parsing.
  - Remove placeholder/dummy `checkpoint = {}` when not using pretrained model; use a typed small dataclass for checkpoint metadata.
  - Replace many global mutable lists (avg_* variables) with a metrics collector object to avoid implicit state leakage.
  - Remove commented-out dead code and clarify which features are experimental vs supported.
  - Add unit tests for:
    - CLI overrides, full_model_export, test_epoch telemetry branch, renormalisation functions, and dataset indexing.

- Documentation / user feedback
  - Add clearer docstrings and parameter validation for public functions (train_epoch, test_epoch, plot_epoch_data, full_model_export).
  - Add a CONTRIBUTING or README note describing required environment, supported precision, and tensorboard/profiler dependencies.

- Misc
  - `tensorboard_logger.launch_dashboard()` is called after TensorBoardLogger already may auto-launch — avoid duplicate launches and guard for headless environments.
  - Ensure `execution_timer` is optional: many calls assume it exists; guard calls when `use_execution_timer` is False.
  - Ensure graceful shutdown when saving profiler/tensorboard resources (check existence before calling .close()).

Suggested next steps
  1. Fix blocking/obvious runtime bugs: undefined `epoch` in test_epoch, full_model_export missing return, inconsistent loss selection checks.
  2. Add unit tests for fixed items.
  3. Triage and schedule refactors: dataset/memory, precision handling, and plotting/telemetry consolidation.













improvements:

#### ~~~~

#### ~~~~

#### ~~~~

#### ~~~~ aLLOW EXECUTION TIMER TO AUTO STOP ALL STARTED TIMER WHEN IT IS SHUTDOWN UNEPECTANTLY???!!!

#### ~~~~ IMPROVE Move o full tensoboard data logging for perf 

#### ~~~~ [DONE!] PArent ID not being set properly as all models pick up 0 for parent id when they are pretrain wheich means thee data was not  ofund for them to nnumbe rit bbut they wer epretrain

#### ~~~~ IMPROVE:     avg_loss_snr.append((10*np.log10(signal_spatial_retention_raw/numof_false_positives_xy)/batch_size) if numof_false_positives_xy !=0 else 1000)  # Avoid div by 0 but needs a better value for this case!

#### ~~~~ [DONE! TEST] FIX SNR Calulation

#### ~~~~ INVESTIGATE: discrepency between detailed performance metrics and the histogram + above/bbelow graphs for recon and detetcions --- THIS MAY BE DUE TO THE ACTUAL TARGET NOT HAVING 300 PIX IN IT AS WE USE THE SINGAAL POINTS USER SETTING IN THE GRAPHS, THE ACTUAL VALUE COULD BE LOWER IF THE TARGET STRATED WITH LESS THAN 300 BEFORE DEGREDATION!!!!
            TURNS OUT EVERYTHING IS FINE EXCEPT FOR THE HORIZONTAL LINES ON THE 'reconstruction telementry per epoch' GRAPH, WHICHI SHOW THE TRUE NUMBER OF SINGAL AND ZERO POINTS, THIS NUMBER IS DERIVED FROM THE USER SETTING I.,E 300 BUT IF THE RAW TRAIN DATA SAMPLE HAD LESS THAN 300 THEN IT COULD HAVE LESS. THE LINE NEEDS TO BE DERIVED FROM THE TRUE VALUE, BUT THIS CHANGES SO WILL NEED TO BE TRACKED IN THE TELEMETRY ARRAY, THE DATA WE NEED CAN BE GETTEN FROM THE DETAILED PERMOFANCE METRICS FUCNTION 'raw_signal_points' VALUE STORED THERE PER EPOCH! calculate the raw nuot singfal piints form this 
                ALTHOUGH EVEN THE IDEA OF THESE VALUES IS WRONG BECAUSE THEY COME AFTER SINGAL DEGREDATION WHICH REMOVES POINTS, SO THE TRUE VALUE IS THE NUMBER OF POINTS IN THE ORIGINAL UNDEGRADED IMAGE, NEED TO TRACK THIS SOMEHOW
#### ~~~~ INVESTIGATE - Issue with the precision setting? seems to be set fixed due to issues?

#### ~~~~ [DONE!] IMPROVMENT: Autodetect counters file on drive and if not found then create from fresh!

#### ~~~~ IMPROVEMENT: update the plots to get more 3D plots and also to show 3D from birds eye view underneat each one

#### ~~~~ IMPROVEMENT: Make a scheduler that allows for input parameter updates at set epoch schedules e.g singal points, etc 

#### ~~~~ FIX the belief telemetry output to be integer as described, make sure all downstream is okay with it not being a tensor anymore!

### ~~~~~ FIX: 'Epoch Times' list not being appended to correctly, only final? epoch time recorded?? (Obbserved taling the mean of the array always rssults in 1/2 of the total processing time)? 

### ~~~~~ FIX: async load the large datlaoder

### ~~~~~ FIX: degradation fucntions, costly and need optimisation

### ~~~~~ INVESTIGATE: Torch profiler is not working, need to investigate, possibly due to not using the when loop 

### ~~~~~ FIX: 3d reconstrucxtion costly and requires reoptimisations

### ~~~~~ FIX: Handle execution timer misiing start/stop data gracefully, do not allow program crash 

### ~~~~~ FIX hook_flush_interval, setting this to greater than 1 results in just th eepoch in question being saved, i.,e setting to 4 saves every fourth epoch rather than stroing each epoch between then and then dumping them all!!

### ~~~~~ FIX TOF std_dev which if applied ends up with values outside of the range 0-time_dimension, either clamp the values to this range, or if values exceed it then remove those points is in practice the std_dev would carry them into previous or post frames rather than current frame. 
            HALF FIXED, now have code to either clamp to frame (unnatural) or clip away, currently hard coded to use latter. But this needs adding as a user parameter and connecting, tricky as it comes from the dataloader that does the degradation now so neeed ot pass as param to dataloader!

### ~~~~~ FIX, [UPDATE: THIS IS AN ERROR WITH ACB_MSE CREATING NANS WHICH THEN ESULT IN 0 VALUES BEING ASSINGED, THE PROGRAM DOES WORK USING STANDARD MSE LOASS AT FP16] when running with precision = 16 an error ocurs in backwards pass saying cannot perfrom backward pass of int, i think its saying the result of the loss is an int?? Investigate

### ~~~~~ INVESTIGATE: Attach the new masking optimised normalisation check if it need a corresponding renorm

### ~~~~~ IMPROVEMENT: Add the new performance metrics per epoch to the history da dictionary to clean up??

### ~~~~~ IMPROVEMENT: Clean up the performance loss plottsing code, it is too long and unwieldy, move it to an external file and load as a function

### ~~~~~ INVESTIGATE: #Noticed we need to update and cleanup all the terminal printing during the visulisations, clean up the weird line spaces and make sure to print what plot is currntly generating as some take a while and the progress bar doesn’t let user know what plot we are currently on

### ~~~~~ IMPROVEMENT: Update the model and data paths to folders inside the root dir so that they do not need be defined, and so that people can doanload the git repo and just press run without setting new paths etc 

### ~~~~~ Update the completed prints to terminal after saving things to only say that if the task is actually completeted, atm it will do it regardless, as in the error on .py save 

### ~~~~~ IMPROVEMENT: Clean up custom autoencoder.py file saver terminal prints left over from debugging

### ~~~~~ FIX: fix epoch numbering printouts? they seem to report 1 epoch greater than they should

### ~~~~~ IMPROVEMNT: Add in automatic Enc/Dec layer size calulations

### ~~~~~ IMPROVEMENT: Allow seperate loss fucntion for testing/validation phase?

### ~~~~~ FIX: Properly track the choices for split loss funcs in txt output file 

### ~~~~~ IMPROVMENT: Explicitlly pass in split_loss_functions to the split custom weigted func atm is not done to simplify the code but is not ideal

### ~~~~~ IMPROVMENT: Create flatten module in main body so noise can be added to the 3D cube rather than slicewise?

### ~~~~~ IMPROVEMNT: colour true signal points red in the input distroted image so that viewer can see the true signal points and the noise added

### ~~~~~ FIX BUG : xlsxwriter.exceptions.InvalidWorksheetName: Excel worksheet name 'T_fullframe_weighting_0.5 direct' must be <= 31 chars.


### REFACTOR: func: quantify_loss_performance~ Appends the average performance metrics for the batch to the corresponding list of average metrics results for each epoch which is tracked eleshwehre in the enhanced performance tracking system and needs simplifying and condensing!!!

### REFACTOR: func: belief_telemetry ~ This function creates histogram plots of the pixel values recorded by the belief telemtry system [which needs renaming and reworking to simplify. (should be reconstruction thresholding telemtetry?)]  whihc records the values directly out of the netwrok before our reconstruction thresholding in the custom renomalisation is applied. This is important to keep ttrack of what is occusring before our iytput processing as it may be hiding errors.

### REFACTOR: func: create_settings_dict ~ Terrible, nneds compleate clean up and overhaul

### REFACTOR: func: plot_detailed_performance_loss etc  ~ PART OF THE LOSS PERFORMANCE SYSTEM NEEDEING OVERHAUL SIMPLIFICATION MODUIOARISATION AND THEN MOVING TO SEPERATE FILE 

### REFACTOR: func:  ~ 

### REFACTOR: func:  ~ 

### REFACTOR: func:  ~ 

### REFACTOR: func:  ~ 

### REFACTOR: func:  ~ 

### REFACTOR: func:  ~ 


### ~~~~~ [DONE!] FIX: Datalaoder calls for every individual sample in the ifnal bacth instead of loading them all at once in a singl eslice with the rang eof index's ... this will require reevaluation of the metho dused to iterate the large datloader as woul dno longe rbe checking index by index ?

### ~~~~~ [DONE!] FIX: INcorrct average epoch time calculation in final summary

### ~~~~~  [DONE!] FIX the way the precision value is saved for model reruns, currently saves the double_precision boolean although have now moved to numerical 'precision' variable that accepts 16. 32 and 64 corresponding to half, single and double 

### ~~~~~ [DONE!] FIX plotting fucntion iterates through test_loader batches choosing first image in each batch, this leads to error if there are less batches than the number to plot selected

#### ~~~~ [DONE!] Fix issue in the time scale of the 3d plots?? seems non linear and also goes to 5000!?

### ~~~~~ [DONE!] Connect the inject_seed sytem up and add an interval

### ~~~~~ [DONE!] USe seeding value to controll data spareness and noise addition so that thay can be set to fixed

### ~~~~~ [DONE!] Create new unseen dataset for performance analysis on hyperparameter optimisation, could have a switch to make it use the val/test data?

### ~~~~~ [DONE!] Record the time taken per epoch, and then alongside the loss vs epoch plot (which dosent show time, so 1 epoch that takes 2 hrs is same as one that takes 1 min) plot loss vs time as a seperate plot

### ~~~~~ [DONE!] Add user controll to overide double precision processing

### ~~~~~ [DONE!] Improve the pixel telemtry per epoch by adding a dotted green line indicating the true number of signal points\

### ~~~~~ [DONE!] Make sure that autoecoder Encoder and Decoder are saved along with model in the models folder 

### ~~~~~ [DONE!] Fix memory leak in testing function loss calulation

### ~~~~~ [DONE!] Investigate and fix memory leak in plotting function

### ~~~~~ [DONE!] Reduce memory usage in loss calulation by removing lists

### ~~~~~ [DONE!] Ground plots and std devs in physical units

### ~~~~~~ [DONE!] Allow normalisation/renorm to be bypassed, to check how it affects results 

### ~~~~~~ [DONE!] Find out what is going on with recon threshold scaling issue

### ~~~~~~ [DONE!] fix noise adding to the data, it is not working as intended, need to retain clean images for label data 

### ~~~~~ [DONE!] MUST SEPERATE 3D recon and flattening from the normalisation and renormalisation

### ~~~~~ [DONE!] Fix Val loss save bug

### ~~~~~ [DONE!] Custom MSE loss fucntion with weighting on zero vals to solve class imbalence

### ~~~~~ [DONE!] Move things to other files (AE, Helper funcs, Visulisations etc)

### ~~~~~ [DONE!] Fix reconstruction threshold, use recon threshold to set bottom limit in custom normalisation

### ~~~~~ [DONE!] Turn plot or save into a function 

### ~~~~~ [DONE!] Add in a way to save the model after each epoch, so that if the program crashes we can still use the last saved model

### ~~~~~ [DONE!] Find way to allow user to exit which is non blocking 

### ~~~~~ [DONE!] Train on labeld data which has the fill line paths in labels and then just points on line in the raw data?

### ~~~~~ [DONE!] change telemetry variable name to output_pixel_telemetry

### ~~~~~ [DONE!] Fix this " if plot_higher_dim: AE_visulisation(en...)" break out all seperate plotting functions
    
### ~~~~~ [DONE!] sort out val, test and train properly

### ~~~~~ [DONE!] Update all layer activation tracking from lists and numpy to torch tensors throughout pipeline for speed

### ~~~~~ [DONE!] Search for and fix errors in custom norm an renorm

### ~~~~~ [DONE!] Seperate and moularise renorm and 3D reconstruction

### ~~~~~ [DONE!] Add all advanced program settings to end of net summary txt file i.e what typ eof normalisation used etc, also add th enam eof the autoencoder file i.e AE_V1 etc from the module name 

### ~~~~~ [DONE!] update custom mse loss fucntion so that user arguments are set in settings page rather than at function def by defualts i.e (zero_weighting=1, nonzero_weighting=5)

### ~~~~~ [DONE!] could investigate programatically setting the non_zero weighting based on the ratio of zero points to non zero points in the data set which would balance out the two classes in the loss functions eyes

### ~~~~~ [DONE!] Add way for program to save the raw data for 3d plots so that they can be replotted after training and reviewed in rotatable 3d 

### ~~~~~ [DONE!] Check if running model in dp (fp64) is causing large slow down???

### ~~~~~ [DONE!] Update noise points to take a range as input and randomly select number for each image from the range

### ~~~~~ [DONE!] Add fcuntion next to noise adder that drops out pixels, then can have the labeld image with high signal points and then dropout the points in the input image to network so as to train it to find dense line from sparse points!

### ~~~~~ [DONE!] Add plots of each individual degradation step rathert than just all shown on one (this could be done instead of the current end of epoch 10 plots or alongside)

### ~~~~~ [DONE!] add masking directly to the trainer so we can see masked output too 

### ~~~~~  [DONE!] Label loss plots y axis programatically based on user loss function selection
