import os

def deleteTmpFiles():
    
    filename_list = ('output/sigmaTDownSample.csv',\
                     'output/reflectance.csv',\
                     'output/reflectanceStderr.csv',\
                     'output/densityMap.csv',\
                     'output/densityMap.csv');
                    
    for filename in filename_list:
        try:
            os.remove(filename);
        except OSError:
            pass
