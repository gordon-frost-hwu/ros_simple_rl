import os

results_dir = "/tmp/testing/"
scriptDir = os.path.dirname(os.path.realpath('__file__'))
print("scriptDir: {0}".format(scriptDir))
filename = "{0}{1}".format(scriptDir, '/stober_cacla_nessie.py') #os.path.join(scriptDir, '/stober_cacla_nessie.py')
print("main script path: {0}".format(filename))
os.system("cp {0} {1}".format(filename, results_dir))
#For accessing the file in the same folder
filename = "same.txt"

#For accessing the file in a folder contained in the current folder
#filename = os.path.join(scriptDir, 'Folder1.1/same.txt')

#For accessing the file in the parent folder of the current folder
filename = os.path.join(scriptDir, '../basis_functions/simple_basis_functions.py')
print("basis functions abs path: {0}".format(filename))
os.system("cp {0} {1}".format(filename, results_dir))
#For accessing the file inside a sibling folder.
#filename = os.path.join(scriptDir, '../Folder2/same.txt')
#filename = os.path.abspath(os.path.realpath(filename))
