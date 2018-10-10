import glob
import os
import sys
import ast

def get_catagories_dir_path(source_dir_path):
    R_CW_HS_dir_path = os.path.join(source_dir_path,"right-clearwin-highscore")
    R_CW_LS_dir_path = os.path.join(source_dir_path,"right-clearwin-lowscore")
    R_NN_HS_dir_path = os.path.join(source_dir_path,"right-neck&neck-highscore")
    R_NN_LS_dir_path = os.path.join(source_dir_path,"right-neck&neck-lowscore")
    W_CW_HS_dir_path = os.path.join(source_dir_path,"wrong-clearwin-highscore")
    W_CW_LS_dir_path = os.path.join(source_dir_path,"wrong-clearwin-lowscore")
    W_NN_HS_dir_path = os.path.join(source_dir_path,"wrong-neck&neck-highscore")
    W_NN_LS_dir_path = os.path.join(source_dir_path,"wrong-neck&neck-lowscore")
    return R_CW_HS_dir_path, R_CW_LS_dir_path, R_NN_HS_dir_path, R_NN_LS_dir_path, W_CW_HS_dir_path, W_CW_LS_dir_path, W_NN_HS_dir_path, W_NN_LS_dir_path

def move_file(file_to_move, source_dir_path,  destination_dir_path):
    if not os.path.exists(destination_dir_path):
        os.makedirs(destination_dir_path)
    os.rename(os.path.join(source_dir_path, file_to_move),
              os.path.join(destination_dir_path, file_to_move))

def word_counter(text, mode="basic"):
    # TODO: Add tokenizer mode; see annotate() func in prepro.py
    if mode=="basic":
        return len(text.rstrip().split(' '))
    else:
        raise Exception("Currently only basic mode is available.")

def catagize_files(source_dir_path):
    # Mutually Exclusive Catagories Directory Path:
    catagories_dir_path = get_catagories_dir_path(source_dir_path)
    R_CW_HS_dir_path = catagories_dir_path[0]
    R_CW_LS_dir_path = catagories_dir_path[1]
    R_NN_HS_dir_path = catagories_dir_path[2]
    R_NN_LS_dir_path = catagories_dir_path[3]
    W_CW_HS_dir_path = catagories_dir_path[4]
    W_CW_LS_dir_path = catagories_dir_path[5]
    W_NN_HS_dir_path = catagories_dir_path[6]
    W_NN_LS_dir_path = catagories_dir_path[7]

    # Basic Catagories Counters:
    #   NOTED:
    #   -(Mutually Exclusive) Base Catagories Counter:
    #       RIGHT_counter = R_CW_HS_counter[0] + R_CW_LS_counter[0] + R_NN_HS_counter[0] + R_NN_LS_counter[0]
    #       WRONG_counter = W_CW_HS_counter[0] + W_CW_LS_counter[0] + W_NN_HS_counter[0] + W_NN_LS_counter[0]
    #       TOTAL_SAMPLES_counter = RIGHT_counter + WRONG_counter
    #
    #   -Subset Catagories Counter:
    #       R_LP_counter= R_CW_HS_counter[1] + R_CW_LS_counter[1] + R_NN_HS_counter[1] + R_NN_LS_counter[1]
    #       W_LP_counter= W_CW_HS_counter[1] + W_CW_LS_counter[1] + W_NN_HS_counter[1] + W_NN_LS_counter[1]
    #       R_LAA_counter= R_CW_HS_counter[2] + R_CW_LS_counter[2] + R_NN_HS_counter[2] + R_NN_LS_counter[2]
    #       W_LAA_counter= W_CW_HS_counter[2] + W_CW_LS_counter[2] + W_NN_HS_counter[2] + W_NN_LS_counter[2]
    #       TOTAL_LongPredict_counter = R_LP_counter + W_LP_counter
    #       TOTAL_LongAnsAll_counter = R_LAA_counter + W_LAA_counter
    #
    #       TOTAL_LongPredict_counter = counter for when top most prediction have >4 words length.
    #       TOTAL_LongAnsAll_counter = counter for when all answers have >4 words length.
    #

    R_CW_HS_counter = [0,0,0]
    R_CW_LS_counter = [0,0,0]
    R_NN_HS_counter = [0,0,0]
    R_NN_LS_counter = [0,0,0]
    W_CW_HS_counter = [0,0,0]
    W_CW_LS_counter = [0,0,0]
    W_NN_HS_counter = [0,0,0]
    W_NN_LS_counter = [0,0,0]

    # Other Catagories Counters:
    #   NOTED:
    #       TOP_IS_RIGHT_counter = counter for when one of the top prediction is correct.
    #       REQUIRE_MANUAL_CATAGORIZES_counter = counter for when require manual catagorizing.
    TOP_IS_RIGHT_counter = 0
    REQUIRE_MANUAL_CATAGORIZES_counter = 0
    
    for file in glob.glob(os.path.join(source_dir_path,"gt_*.txt")):
        filename, _ = os.path.basename(file).split('.', 1)
        txt_file_to_move = filename+".txt"
        png_file_to_move = filename+".png"

        with open(file, 'r') as f:
            try:
                lines = f.read().splitlines()
                top_predictions = [lines[i].split('\t', 2)[2] for i in range(1,6)]
                top_scores = [float(lines[i].split('\t', 2)[1]) for i in range(1,6)]
            except Exception as e:
                print("ERROR:{0}\n\tFILE:{1}\n\tLINES:{2}".format(e,f,lines))
                REQUIRE_MANUAL_CATAGORIZES_counter += 1
                continue
            
            best_predict = top_predictions[0]
            best_score = top_scores[0]
            next_best_predict = top_predictions[1]
            next_best_score = top_scores[1]
            if lines[7].startswith("\t"):
                lines[7] = lines[7][1:]
            if lines[7].startswith("[") and lines[7].endswith("]"):
                targets_from_data = ast.literal_eval(lines[7])
            else:
                targets_from_data = [lines[7]]
            
            is_right = (best_predict in targets_from_data)
            is_clearwin = (best_score > 1.5*next_best_score)
            is_highscore = (best_score >= 0.5)
            is_in_top = (not set(targets_from_data).isdisjoint(top_predictions))
            is_longPredict = (word_counter(best_predict) > 4)
            is_longAnsAll = any(word_counter(ans) > 4 for ans in targets_from_data)
            #import pdb;pdb.set_trace()
    

        if is_in_top:
            TOP_IS_RIGHT_counter += 1
        
        if is_right and is_clearwin and is_highscore:
            R_CW_HS_counter[0] += 1
            if is_longPredict:
                R_CW_HS_counter[1] += 1
            if is_longAnsAll:
                R_CW_HS_counter[2] += 1
            move_file(txt_file_to_move, source_dir_path, R_CW_HS_dir_path)
            move_file(png_file_to_move, source_dir_path, R_CW_HS_dir_path)
        
        elif is_right and is_clearwin and not is_highscore:
            R_CW_LS_counter[0] += 1
            if is_longPredict:
                R_CW_LS_counter[1] += 1
            if is_longAnsAll:
                R_CW_LS_counter[2] += 1
            move_file(txt_file_to_move, source_dir_path, R_CW_LS_dir_path)
            move_file(png_file_to_move, source_dir_path, R_CW_LS_dir_path)

        elif is_right and not is_clearwin and is_highscore:
            R_NN_HS_counter[0] += 1
            if is_longPredict:
                R_NN_HS_counter[1] += 1
            if is_longAnsAll:
                R_NN_HS_counter[2] += 1
            move_file(txt_file_to_move, source_dir_path, R_NN_HS_dir_path)
            move_file(png_file_to_move, source_dir_path, R_NN_HS_dir_path)

        elif is_right and not is_clearwin and not is_highscore:
            R_NN_LS_counter[0] += 1
            if is_longPredict:
                R_NN_LS_counter[1] += 1
            if is_longAnsAll:
                R_NN_LS_counter[2] += 1
            move_file(txt_file_to_move, source_dir_path, R_NN_LS_dir_path)
            move_file(png_file_to_move, source_dir_path, R_NN_LS_dir_path)

        elif not is_right and is_clearwin and is_highscore:
            W_CW_HS_counter[0] += 1
            if is_longPredict:
                W_CW_HS_counter[1] += 1
            if is_longAnsAll:
                W_CW_HS_counter[2] += 1
            move_file(txt_file_to_move, source_dir_path, W_CW_HS_dir_path)
            move_file(png_file_to_move, source_dir_path, W_CW_HS_dir_path)

        elif not is_right and is_clearwin and not is_highscore:
            W_CW_LS_counter[0] += 1
            if is_longPredict:
                W_CW_LS_counter[1] += 1
            if is_longAnsAll:
                W_CW_LS_counter[2] += 1
            move_file(txt_file_to_move, source_dir_path, W_CW_LS_dir_path)
            move_file(png_file_to_move, source_dir_path, W_CW_LS_dir_path)

        elif not is_right and not is_clearwin and is_highscore:
            W_NN_HS_counter[0] += 1
            if is_longPredict:
                W_NN_HS_counter[1] += 1
            if is_longAnsAll:
                W_NN_HS_counter[2] += 1
            move_file(txt_file_to_move, source_dir_path, W_NN_HS_dir_path)
            move_file(png_file_to_move, source_dir_path, W_NN_HS_dir_path)
        else:
            W_NN_LS_counter[0] += 1
            if is_longPredict:
                W_NN_LS_counter[1] += 1
            if is_longAnsAll:
                W_NN_LS_counter[2] += 1
            move_file(txt_file_to_move, source_dir_path, W_NN_LS_dir_path)
            move_file(png_file_to_move, source_dir_path, W_NN_LS_dir_path)
                    
    RIGHT_counter = R_CW_HS_counter[0] + R_CW_LS_counter[0] + R_NN_HS_counter[0] + R_NN_LS_counter[0]
    WRONG_counter = W_CW_HS_counter[0] + W_CW_LS_counter[0] + W_NN_HS_counter[0] + W_NN_LS_counter[0]
    R_LP_counter= R_CW_HS_counter[1] + R_CW_LS_counter[1] + R_NN_HS_counter[1] + R_NN_LS_counter[1]
    W_LP_counter= W_CW_HS_counter[1] + W_CW_LS_counter[1] + W_NN_HS_counter[1] + W_NN_LS_counter[1]
    R_LAA_counter= R_CW_HS_counter[2] + R_CW_LS_counter[2] + R_NN_HS_counter[2] + R_NN_LS_counter[2]
    W_LAA_counter= W_CW_HS_counter[2] + W_CW_LS_counter[2] + W_NN_HS_counter[2] + W_NN_LS_counter[2]
    TOTAL_SAMPLES_counter = RIGHT_counter + WRONG_counter
    TOTAL_LongPredict_counter = R_LP_counter + W_LP_counter
    TOTAL_LongAnsAll_counter = R_LAA_counter + W_LAA_counter

    
    print("MUTUALLY EXCLUSIVE CATAGORIES: ")
    print(" right_prediction\t\t{0} ({1}%)\t(long_prediction={2}; all_long_ans={3})".format(RIGHT_counter,
                                                                                            100*RIGHT_counter/TOTAL_SAMPLES_counter,
                                                                                            R_LP_counter,
                                                                                            R_LAA_counter))
    print("   right-clearwin-highscore\t{0}\t\t(long_prediction={1}; all_long_ans={2})".format(R_CW_HS_counter[0],
                                                                                               R_CW_HS_counter[1],
                                                                                               R_CW_HS_counter[2]))
    print("   right-clearwin-lowscore\t{0}\t\t(long_prediction={1}; all_long_ans={2})".format(R_CW_LS_counter[0],
                                                                                              R_CW_LS_counter[1],
                                                                                              R_CW_LS_counter[2]))
    print("   right-neck&neck-highscore\t{0}\t\t(long_prediction={1}; all_long_ans={2})".format(R_NN_HS_counter[0],
                                                                                                R_NN_HS_counter[1],
                                                                                                R_NN_HS_counter[2]))
    print("   right-neck&neck-lowscore\t{0}\t\t(long_prediction={1}; all_long_ans={2})".format(R_NN_LS_counter[0],
                                                                                               R_NN_LS_counter[1],
                                                                                               R_NN_LS_counter[2]))
                                                                                             
    print(" wrong_prediction\t\t{0} ({1}%)\t(long_prediction={2}; all_long_ans={3})".format(WRONG_counter,
                                                                                            100*WRONG_counter/TOTAL_SAMPLES_counter,
                                                                                            W_LP_counter,
                                                                                            W_LAA_counter))
    print("   wrong-clearwin-highscore\t{0}\t\t(long_prediction={1}; all_long_ans={2})".format(W_CW_HS_counter[0],
                                                                                               W_CW_HS_counter[1],
                                                                                               W_CW_HS_counter[2]))
    print("   wrong-clearwin-lowscore\t{0}\t\t(long_prediction={1}; all_long_ans={2})".format(W_CW_LS_counter[0],
                                                                                              W_CW_LS_counter[1],
                                                                                              W_CW_LS_counter[2]))
    print("   wrong-neck&neck-highscore\t{0}\t\t(long_prediction={1}; all_long_ans={2})".format(W_NN_HS_counter[0],
                                                                                                W_NN_HS_counter[1],
                                                                                                W_NN_HS_counter[2]))
    print("   wrong-neck&neck-lowscore\t{0}\t\t(long_prediction={1}; all_long_ans={2})".format(W_NN_LS_counter[0],
                                                                                               W_NN_LS_counter[1],
                                                                                               W_NN_LS_counter[2]))

    print(" total_samples\t\t\t{0}\t\t(long_prediction={1}; all_long_ans={2})".format(TOTAL_SAMPLES_counter,
                                                                                      TOTAL_LongPredict_counter,
                                                                                      TOTAL_LongAnsAll_counter))

    print("\nOTHER CATAGORIES:")
    print(" right_among_top_predictions\t{0} ({1}%)".format(TOP_IS_RIGHT_counter, 100*TOP_IS_RIGHT_counter/TOTAL_SAMPLES_counter))
    print(" require_manual_catagorize:\t{0}".format(REQUIRE_MANUAL_CATAGORIZES_counter))


def reset_catagized_files(source_dir_path):
    # Mutually Exclusive Catagories Directory Path:
    catagories_dir_path = get_catagories_dir_path(source_dir_path)
    
    # Move all catagorized files back to original source_dir_path.
    for catagories_dir in catagories_dir_path:
        if os.path.exists(catagories_dir):
            for file in os.listdir(catagories_dir):
                move_file(file, catagories_dir,  source_dir_path)
            os.rmdir(catagories_dir)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise Exception("Only take 2 command line arg: \
                        sys.argv[1] is either 'catagorize' or 'reset'; \
                        sys.argv[2] is the path of dir to be categorize or reset.")

    if sys.argv[1] == "catagorize":
        dir_tobe_catagorize = str(sys.argv[2])
        catagize_files(dir_tobe_catagorize)
    elif sys.argv[1] == "reset":
        dir_tobe_reset = str(sys.argv[2])
        reset_catagized_files(dir_tobe_reset)
    else:
        raise Exception("sys.argv[1] is either 'catagorize' or 'reset'.")




