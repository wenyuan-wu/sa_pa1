hyp_lines = open("hyp.txt").readlines()
ref_lines = open("ref.txt").readlines()

correct_ngrams = 0
all_ngrams = 0

for ref_line, hyp_line in zip(ref_lines, hyp_lines):
    correct_for_this_pair, all_for_this_pair = compute_correct_and_all(ref_line, hyp_line)
    correct_ngrams += correct_for_this_pair
    all_ngrams += all_for_this_pair

precison = correct_ngrams / all_ngrams
