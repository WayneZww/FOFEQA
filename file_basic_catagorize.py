import glob
import os


def move_file(file_to_move, source_dir_path,  destination_dir_path):
    if not os.path.exists(destination_dir_path):
        os.makedirs(destination_dir_path)
    os.rename(os.path.join(source_dir_path, file_to_move),
              os.path.join(destination_dir_path, file_to_move))

def catagize_files(source_dir_path):
    # Mutually Exclusive Catagories Directory Path:
    R_CW_HS_dir_path = os.path.join(source_dir_path,"right-clearwin-highscore")
    R_CW_LS_dir_path = os.path.join(source_dir_path,"right-clearwin-lowscore")
    R_NN_HS_dir_path = os.path.join(source_dir_path,"right-neck&neck-highscore")
    R_NN_LS_dir_path = os.path.join(source_dir_path,"right-neck&neck-lowscore")
    W_CW_HS_dir_path = os.path.join(source_dir_path,"wrong-clearwin-highscore")
    W_CW_LS_dir_path = os.path.join(source_dir_path,"wrong-clearwin-lowscore")
    W_NN_HS_dir_path = os.path.join(source_dir_path,"wrong-neck&neck-highscore")
    W_NN_LS_dir_path = os.path.join(source_dir_path,"wrong-neck&neck-lowscore")
    BUG_SAMPLES_dir_path = os.path.join(source_dir_path,"bug_samples")
    
    # Mutually Exclusive Catagories Counters:
    #   NOTED:
    #       RIGHT_counter = R_CW_HS_counter + R_CW_LS_counter + R_NN_HS_counter + R_NN_LS_counter
    #       WRONG_counter = W_CW_HS_counter + W_CW_LS_counter + W_NN_HS_counter + W_NN_LS_counter
    #       TOTAL_SAMPLES_counter = RIGHT_counter + WRONG_counter + BUG_SAMPLES_counter
    R_CW_HS_counter = 0
    R_CW_LS_counter = 0
    R_NN_HS_counter = 0
    R_NN_LS_counter = 0
    W_CW_HS_counter = 0
    W_CW_LS_counter = 0
    W_NN_HS_counter = 0
    W_NN_LS_counter = 0
    BUG_SAMPLES_counter = 0

    # Other Catagories Counters:
    TOP_IS_RIGHT_counter = 0

    for file in glob.glob(os.path.join(source_dir_path,"gt_*.txt")):
        filename, _ = os.path.basename(file).split('.', 1)
        txt_file_to_move = filename+".txt"
        png_file_to_move = filename+".png"

        with open(file, 'r') as f:
            lines = f.read().splitlines()
            top_predictions = [lines[i].split('\t', 2)[2] for i in range(1,6)]
            top_scores = [float(lines[i].split('\t', 2)[1]) for i in range(1,6)]
            
            best_predict = top_predictions[0]
            best_score = top_scores[0]
            next_best_predict = top_predictions[1]
            next_best_score = top_scores[1]
            _, target_in_training = lines[7].split('\t', 1)
            _, target_from_data = lines[9].split('\t', 1)
            
            is_bug = (target_from_data != target_in_training)
            is_right = (best_predict == target_from_data)
            is_clearwin = (best_score > 1.5*next_best_score)
            is_highscore = (best_score >= 0.5)
            is_in_top = (target_from_data in top_predictions)
    
        if is_in_top:
            TOP_IS_RIGHT_counter += 1
        if is_bug:
            BUG_SAMPLES_counter += 1
            move_file(txt_file_to_move, source_dir_path, BUG_SAMPLES_dir_path)
            move_file(png_file_to_move, source_dir_path, BUG_SAMPLES_dir_path)
        else:
            if is_right and is_clearwin and is_highscore:
                R_CW_HS_counter += 1
                move_file(txt_file_to_move, source_dir_path, R_CW_HS_dir_path)
                move_file(png_file_to_move, source_dir_path, R_CW_HS_dir_path)
            
            elif is_right and is_clearwin and not is_highscore:
                R_CW_LS_counter += 1
                move_file(txt_file_to_move, source_dir_path, R_CW_LS_dir_path)
                move_file(png_file_to_move, source_dir_path, R_CW_LS_dir_path)

            elif is_right and not is_clearwin and is_highscore:
                R_NN_HS_counter += 1
                move_file(txt_file_to_move, source_dir_path, R_NN_HS_dir_path)
                move_file(png_file_to_move, source_dir_path, R_NN_HS_dir_path)

            elif is_right and not is_clearwin and not is_highscore:
                R_NN_LS_counter += 1
                move_file(txt_file_to_move, source_dir_path, R_NN_LS_dir_path)
                move_file(png_file_to_move, source_dir_path, R_NN_LS_dir_path)

            elif not is_right and is_clearwin and is_highscore:
                W_CW_HS_counter += 1
                move_file(txt_file_to_move, source_dir_path, W_CW_HS_dir_path)
                move_file(png_file_to_move, source_dir_path, W_CW_HS_dir_path)

            elif not is_right and is_clearwin and not is_highscore:
                W_CW_LS_counter += 1
                move_file(txt_file_to_move, source_dir_path, W_CW_LS_dir_path)
                move_file(png_file_to_move, source_dir_path, W_CW_LS_dir_path)

            elif not is_right and not is_clearwin and is_highscore:
                W_NN_HS_counter += 1
                move_file(txt_file_to_move, source_dir_path, W_NN_HS_dir_path)
                move_file(png_file_to_move, source_dir_path, W_NN_HS_dir_path)
            else:
                W_NN_LS_counter += 1
                move_file(txt_file_to_move, source_dir_path, W_NN_LS_dir_path)
                move_file(png_file_to_move, source_dir_path, W_NN_LS_dir_path)
                    
    RIGHT_counter = R_CW_HS_counter + R_CW_LS_counter + R_NN_HS_counter + R_NN_LS_counter
    WRONG_counter = W_CW_HS_counter + W_CW_LS_counter + W_NN_HS_counter + W_NN_LS_counter
    TOTAL_SAMPLES_counter = RIGHT_counter + WRONG_counter + BUG_SAMPLES_counter
    
    print("MUTUALLY EXCLUSIVE CATAGORIES: ")
    print(" right_prediction\t\t{0} ({1}%)".format(RIGHT_counter, 100*RIGHT_counter/TOTAL_SAMPLES_counter))
    print("   right-clearwin-highscore\t{0}".format(R_CW_HS_counter))
    print("   right-clearwin-lowscore\t{0}".format(R_CW_LS_counter))
    print("   right-neck&neck-highscore\t{0}".format(R_NN_HS_counter))
    print("   right-neck&neck-lowscore\t{0}".format(R_NN_LS_counter))


    print(" wrong_prediction\t\t{0} ({1}%)".format(WRONG_counter, 100*WRONG_counter/TOTAL_SAMPLES_counter))
    print("   wrong-clearwin-highscore\t{0}".format(W_CW_HS_counter))
    print("   wrong-clearwin-lowscore\t{0}".format(W_CW_LS_counter))
    print("   wrong-neck&neck-highscore\t{0}".format(W_NN_HS_counter))
    print("   wrong-neck&neck-lowscore\t{0}".format(W_NN_LS_counter))

    print(" bug_samples\t\t\t{0} ({1}%)".format(BUG_SAMPLES_counter, 100*BUG_SAMPLES_counter/TOTAL_SAMPLES_counter))
    print(" total_samples\t\t\t{0}".format(TOTAL_SAMPLES_counter))

    print("\nOTHER CATAGORIES:")
    print(" right_among_top_predictions\t{0} ({1}%)".format(TOP_IS_RIGHT_counter, 100*TOP_IS_RIGHT_counter/TOTAL_SAMPLES_counter))


if __name__ == "__main__":
    source_dir_path = "/Users/Work/CSE/iNCML/FOFEQA/FOFEQA_SED/models_n_logs/v3_opt4_DrawScore_TestEp51_2018Aug03_21h13m08s__a0.8_mcl16_sn0_ctx5"
    catagize_files(source_dir_path)




