This file contains an overview of the structure and content of your download request from the ESO Science Archive.


For every downloaded dataset, files are listed below with the following structure:

dataset_name
        - archive_file_name (technical name, as saved on disk)	original_file_name (user name, contains relevant information) category size


Please note that, depending on your operating system and method of download, at download time the colons (:) in the archive_file_name as listed below may be replaced by underscores (_).


In order to rename the files on disk from the technical archive_file_name to the more meaningful original_file_name, run the following shell command:
    cat THIS_FILE | awk '$2 ~ /^ADP/ {print "test -f",$2,"&& mv",$2,$3}' | sh


In case you have requested cutouts, the file name on disk contains the TARGET name that you have provided as input. To order files by it when listing them, run the following shell command:
    cat THIS_FILE | awk '$2 ~ /^ADP/ {print $2}' | sort -t_ -k3,3


Your feedback regarding the data quality of the downloaded data products is greatly appreciated. Please contact the ESO Archive Science Group via https://support.eso.org/ , subject: Phase 3 ... thanks!

The download includes contributions from the following collections:
Ref(0)	UVES	https://doi.eso.org/10.18727/archive/50	IDP_UVES_ECH_release_description_v3.1_2023-01-09.pdf	https://www.eso.org/rm/api/v1/public/releaseDescriptions/163

Publications based on observations collected at ESO telescopes must acknowledge this fact (please see: http://archive.eso.org/cms/eso-data-access-policy.html#acknowledgement). In particular, please include a reference to the corresponding DOI(s). They are listed in the third column in the table above and referenced below for each dataset. The following shell command lists them:

	cat THIS_FILE | awk -F/ '$1 ~ /^Ref\(/ {print $0,$NF}' | awk '{print $2, $3}' | sort | uniq


Each collection is described in detail in the corresponding Release Description. They can be downloaded with the following shell command:

	cat THIS_FILE | awk -F/ '$1 ~ /^Ref\(/ {print $0,$NF}' | awk '{printf("curl -o %s_%s %s\n", $6, $4, $5)}' | sh

ADP.2020-06-19T08:52:52.383 Ref(0)
	- ADP.2020-06-19T08:52:52.383.fits	UV_SFLX_462243_2010-06-18T04:13:35.291_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3862080
ADP.2020-06-19T09:16:12.605 Ref(0)
	- ADP.2020-06-19T09:16:12.605.fits	UV_SFLX_462237_2010-07-11T03:42:25.725_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3847680
ADP.2020-06-19T09:17:59.099 Ref(0)
	- ADP.2020-06-19T09:17:59.099.fits	UV_SFLX_462240_2010-07-12T05:21:43.765_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3859200
ADP.2020-06-19T09:12:13.104 Ref(0)
	- ADP.2020-06-19T09:12:13.104.fits	UV_SFLX_462236_2010-07-07T01:23:30.131_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3847680
ADP.2020-06-19T08:40:21.544 Ref(0)
	- ADP.2020-06-19T08:40:21.544.fits	UV_SFLX_462239_2010-06-08T03:31:24.360_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3859200
ADP.2020-06-19T09:23:38.807 Ref(0)
	- ADP.2020-06-19T09:23:38.807.fits	UV_SFLX_462241_2010-07-16T04:57:30.605_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3856320
ADP.2020-06-19T07:46:26.479 Ref(0)
	- ADP.2020-06-19T07:46:26.479.fits	UV_SFLX_462231_2010-04-11T04:40:56.741_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3859200
ADP.2020-06-19T08:01:38.079 Ref(0)
	- ADP.2020-06-19T08:01:38.079.fits	UV_SFLX_462243_2010-05-08T01:27:28.276_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3856320
ADP.2020-06-19T09:20:27.421 Ref(0)
	- ADP.2020-06-19T09:20:27.421.fits	UV_SFLX_462234_2010-07-14T05:17:04.629_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3844800
ADP.2020-06-19T07:43:35.385 Ref(0)
	- ADP.2020-06-19T07:43:35.385.fits	UV_SFLX_462242_2010-04-03T09:12:33.272_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3859200
ADP.2020-06-19T08:04:32.130 Ref(0)
	- ADP.2020-06-19T08:04:32.130.fits	UV_SFLX_462238_2010-05-10T09:17:23.911_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3859200
ADP.2020-06-19T09:06:30.275 Ref(0)
	- ADP.2020-06-19T09:06:30.275.fits	UV_SFLX_462234_2010-06-29T05:56:32.105_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3853440
ADP.2020-06-19T08:06:17.481 Ref(0)
	- ADP.2020-06-19T08:06:17.481.fits	UV_SFLX_462233_2010-05-11T03:54:53.659_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3862080
ADP.2020-06-19T09:07:25.683 Ref(0)
	- ADP.2020-06-19T09:07:25.683.fits	UV_SFLX_462235_2010-07-03T02:09:31.306_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3862080
ADP.2020-06-19T09:21:44.398 Ref(0)
	- ADP.2020-06-19T09:21:44.398.fits	UV_SFLX_462238_2010-07-15T02:15:14.204_RED564d1_1x1_08.fits	SCIENCE.SPECTRUM	3856320
