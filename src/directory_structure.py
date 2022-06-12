# coding: utf-8
# File to specify directory structure for project and to specify other methods
# related to reading and writin directories and removing file extensions.

import os


def filesInDirectory(extension, directory):
    '''
    return the list of files in the directory with a specific extension

    Args:
            extension (str): file type to get

            directory (str): path of where files are present

    Returns:
            l (list): list of file names with extension (in current directory)
    '''
    l = []

    # get all file names in current directory
    file_names = os.listdir(directory)

    for file in file_names:
        if file.endswith(extension):
            l.append(file)

    return l


def removeFileExtension(file):
    '''
    remove extension of file passed in as a string

    Args:
            file (str): name of file with extension

    Returns:
            (str): name of file without extension
    '''
    return os.path.splitext(file)[0]


def getWriteDirectory(directory_name, subdirectory_name):
    '''
    get path of directory name specified where information needs
    to be written to (subdirectory specification is optional)

    Args:
            directory_name (str): name of directory to read from

            subdirectory (str): subdirectory of directory specified

    Returns:
            wr_dir (str): path of directory to write data to
    '''

    if subdirectory_name is None:
        wr_dir = os.getcwd() + '/' + directory_name + '/'
    else:
        if subdirectory_name == '/':
            wr_dir = os.getcwd() + '/' + directory_name + '/' + '_' + '/'
        else:
            wr_dir = os.getcwd() + '/' + directory_name + \
                '/' + subdirectory_name + '/'

    print("Writing to directory: " + wr_dir)

    os.makedirs(wr_dir, exist_ok=True)
    return wr_dir


def getAllSubfoldersOfFolder(path):
    '''
    get all folders in the path specified

    Args;
            path (str): path where files are

    Returns:
            (list): list of strings with all folder names 
    '''
    return [dI for dI in os.listdir(path) if os.path.isdir(os.path.join(path, dI))]
