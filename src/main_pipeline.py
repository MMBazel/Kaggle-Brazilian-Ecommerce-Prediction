# Structure of this script:
# Create a global performance dictionary (so all steps are logged)
# Provide arguments when running python script
# Every run gets added, along with the arguments provided (ex: different CPU setups)

# This script should
# 1. Grab the full path of where the program is being run


# 2. Run the following pipelines
#  a. Dataloader & prepper
#   i. Read files
#   ii. Unzip files
#   iii. Preprocess files (including casting datatypes)


#  b. Featurization
#   i. Create features
#   ii. Process holiday features
#   iii. Create SQL context
#   iv. Create customer holiday features
#   v. Create labels


#  c. Training
#   i. Create training & test sets
#   ii. Train model
#   iii. Pickle model


import sys
import getopt
from datetime import datetime


def specify_CPU_run(argv):
    current_datetime = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    arg_computer = ""
    arg_user = ""
    arg_version = ""
    arg_print = ""
    arg_help = "{0} -c <computer> -u <user> -v <version> -p <print>".format(argv[0])

    try:
        opts, args = getopt.getopt(
            argv[1:], "hc:u:v:p:", ["help", "computer=", "user=", "version=", "print="]
        )
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)
            sys.exit()
        elif opt in ("-c", "--computer"):
            arg_computer = arg
        elif opt in ("-u", "--user"):
            arg_user = arg
        elif opt in ("-v", "--version"):
            arg_version = arg
        elif opt in ("-p", "--print"):
            arg_print = arg

    if arg_print == "True":
        print("computer:", arg_computer)
        print("user:", arg_user)
        print("version:", arg_version)
        print("current datetime:", current_datetime)
        print("print:", arg_print)

    return arg_print


if __name__ == "__main__":
    specify_CPU_run(sys.argv)
