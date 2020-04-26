#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import Part1, Part2, Part4, Part5, Part6
import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='HW1 Documentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     epilog="Example:\npython q3main.py -r Dataset -tr q2_train_set.txt -te q2_test_set.txt -g q2_gag_sequence.txt\nPLEASE CLOSE FIGURES TO CONTINUE EXECUTION...")
    parser.add_argument("-r", "--root",metavar= 'rootfile', type=str, default='./Dataset', help= "Please enter the root folder path")
    parser.add_argument("-tr", "--train", metavar= 'trainfile', type=str, default='q2_train_set.txt', help= "Please enter the train file name")
    parser.add_argument("-te", "--test", metavar= 'testfile', type=str, default='q2_test_set.txt', help= "Please enter the test file name")
    parser.add_argument("-g", "--gag", metavar= 'gagfile', type=str, default='q2_gag_sequence.txt', help= "Please enter the gag sequence file name")
    args = parser.parse_args()
    #Part 1
    print("Part 1")
    Part1.part1(args.root, args.train, args.test)
    input("Press Enter to continue...")
    #Part 2
    print("Part 2&3")
    Part2.part2(args.root, args.train, args.gag)
    input("Press Enter to continue...")
    #Part 4
    print("Part 4\nPlease wait while drawing graphs...")
    Part4.part4(args.root, args.train, args.test)
    #Part 5
    print("Part 5")
    Part5.part5(args.root, args.train, args.test)
    #Part 6
    print("Part 6")
    Part6.part6(args.root, args.train)
    print("Program finished")    
main()