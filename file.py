import os
s = os.sep
root = ".." + s
for i in os.listdir(root):
    if os.path.isfile(os.path.join(root,i)):
    	print os.path.join(root,i)
        print i