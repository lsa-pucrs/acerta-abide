### get_s3_paths.py
#
# Written by Daniel Clark, Child Mind Institute, 2014
#
# Script for downloading ABIDE preprocessed data from AWS S3 bucket.
# This requires that the phenotypic data file that is located at:
#    https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed1.csv
# be copied locally.

# Import packages
import csv
import os
import pandas
import urllib, urllib2
import hashlib
import re

# Path to phenotypic csv file
csv_path = './Phenotypic_V1_0b_preprocessed1.csv'
# Download directory
download_root = '.'
# Choose pipeline, strategy, and derivative of interest
pipeline = 'cpac'
strategy = 'filt_global'
derivatives = ['func_preproc']

# S3 path prefix
s3_prefix = 'https://s3.amazonaws.com/fcp-indi/data/Projects/'\
            'ABIDE_Initiative/Outputs/'

# --- Download from the S3 bucket based on phenotypic conditions ---
# Read in phenotypic file csv
sub_pheno_list = []
csv_in = pandas.read_csv(csv_path)
r = 0
# Iterate through the csv rows (subjects)
for row in csv_in.iterrows():
    site = csv_in['SITE_ID'][r]
    sub_id = csv_in['SUB_ID'][r]
    file_id = csv_in['FILE_ID'][r]
    sex = csv_in['SEX'][r]
    age = csv_in['AGE_AT_SCAN'][r]
    # Test phenotypic conditions
    # if (site == 'CALTECH' and sex == 1 and age > 30):
    sub_pheno_list.append(file_id)
    r += 1

# Strip out 'no_filename's from list
sub_pheno_list = [s for s in sub_pheno_list if s != 'no_filename']

class HeadRequest(urllib2.Request):
    def get_method(self):
        return "HEAD"

def download(file_path, download_file):
    print file_path, ':',
    if os.path.exists(download_file):
        fhash = hashlib.md5(open(download_file, 'rb').read()).hexdigest()
        try:
            request = HeadRequest(s3_prefix + file_path)
            response = urllib2.urlopen(request)
            response_headers = response.info()
            etag = re.sub(r'^"|"$', '', response_headers['etag'])
            if etag != fhash:
                os.remove(download_file)
            else:
                print 'Match'
        except urllib2.HTTPError, e:
            print ("Error code: %s" % e.code)

    if not os.path.exists(download_file):
        print('Downloading')
        try:
            urllib.urlretrieve(s3_prefix + file_path, download_file)
        except:
            pass
        download(file_path, download_file)


# Fetch s3 path for each file_id that met the phenotypic conditions
path_list = []
for file_id in sub_pheno_list:

    for derivative in derivatives:

        file_path = pipeline + '/' + strategy + '/' + derivative + \
                  '/' + file_id + '_' + derivative + '.nii.gz'
        download_file = os.path.join(download_root, file_path)
        download_dir = os.path.dirname(download_file)

        if not os.path.exists(download_dir):
            os.makedirs(download_dir)

        download(file_path, download_file)