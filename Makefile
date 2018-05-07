# Makefile for MapReduce Page Rank project.

# Customize these paths for your environment.
# -----------------------------------------------------------
spark.root=/usr/local/spark
jar.name=MR_Project_spark-0.0.1-SNAPSHOT.jar
jar.path=target/${jar.name}
job.name=BrainScans
job.name2=Predictions
app.name=BrainScans
app.name2=Eval
local.input=/home/ramkishan/input
local.input2=/home/ramkishan/input2
local.output=/home/ramkishan/output
local.test=/home/ramkishan/test
local.model=/home/ramkishan/model
local.log=/home/ramkishan/log
local.unzipped=/media/ramkishan/Windows/MapReduce/upload
# AWS EMR Execution
aws.release.label=emr-5.11.1
aws.region=us-east-2
aws.bucket.name=ram.cs6240
aws.subnet.id=subnet-345cfc5c
aws.input=input
aws.input2=input2
aws.test=test
aws.model=model
aws.output=output
aws.log=log
aws.log.dir=log
aws.num.nodes=10
aws.instance.type=m4.large
# -----------------------------------------------------------

# Compiles code and builds jar (with dependencies).
jar:
	mvn clean package

# Removes local model directory
clean-local-model:
	rm -rf ${local.model}*

# Removes local output directory.
clean-local-output:
	rm -rf ${local.output}*

# Runs standalone
# Make sure Hadoop  is set up (in /etc/hadoop files) for standalone operation (not pseudo-cluster).
# https://hadoop.apache.org/docs/current/hadoop-project-dist/hadoop-common/SingleCluster.html#Standalone_Operation
alone: clean-local-output
	${spark.root}/bin/spark-submit --class ${job.name} --name "${app.name}" ${jar.path} ${local.input} ${local.output} ${local.test} ${local.model}


alone2: clean-local-output
	${spark.root}/bin/spark-submit --class ${job.name2} --name "${app.name2}" ${jar.path} ${local.model} ${local.output} ${local.test}
	
# Create S3 bucket.
make-bucket:
	aws s3 mb s3://${aws.bucket.name}

# Upload data to S3 input dir.
upload-input-aws:
	aws s3 sync ${local.input} s3://${aws.bucket.name}/${aws.input}
	
# Upload data to S3 input dir.
upload-input2-aws:
	aws s3 sync ${local.input2} s3://${aws.bucket.name}/${aws.input2}
	
# Upload data to S3 input dir.
upload-unzipped-aws:
	aws s3 sync ${local.unzipped} s3://${aws.bucket.name}/${aws.input}	
	
# Upload data to S3 test dir.
upload-test-aws:
	aws s3 sync ${local.test} s3://${aws.bucket.name}/${aws.test}

# Upload models to S3 model dir.
upload-model-aws:
	aws s3 sync ${local.model} s3://${aws.bucket.name}/${aws.model}
	
# Delete S3 output dir.
delete-output-aws:
	aws s3 rm s3://${aws.bucket.name}/ --recursive --exclude "*" --include "${aws.output}*"

# Upload application to S3 bucket.
upload-app-aws:
	aws s3 cp ${jar.path} s3://${aws.bucket.name}

# Main EMR launch.
cloud: delete-output-aws
	aws emr create-cluster \
		--name "Brain scans - 2BT, all train, 1 test" \
		--release-label ${aws.release.label} \
		--instance-groups '[{"InstanceCount":${aws.num.nodes},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
	    --applications Name=Hadoop Name=Spark \
		--steps Type=CUSTOM_JAR,Name="${app.name}",Jar="command-runner.jar",ActionOnFailure=TERMINATE_CLUSTER,Args=["spark-submit","--deploy-mode","cluster","--class","${job.name}","s3://${aws.bucket.name}/${jar.name}","s3://${aws.bucket.name}/${aws.input}","s3://${aws.bucket.name}/${aws.output}","s3://${aws.bucket.name}/${aws.test}","s3://${aws.bucket.name}/${aws.model}"] \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--service-role EMR_DefaultRole \
		--ec2-attributes InstanceProfile=EMR_EC2_DefaultRole,SubnetId=${aws.subnet.id} \
		--region ${aws.region} \
		--enable-debugging \
		--auto-terminate


cloud2: delete-output-aws
	aws emr create-cluster \
		--name "Making predictions" \
		--release-label ${aws.release.label} \
		--instance-groups '[{"InstanceCount":${aws.num.nodes},"InstanceGroupType":"CORE","InstanceType":"${aws.instance.type}"},{"InstanceCount":1,"InstanceGroupType":"MASTER","InstanceType":"${aws.instance.type}"}]' \
	    --applications Name=Hadoop Name=Spark \
		--steps Type=CUSTOM_JAR,Name="${app.name2}",Jar="command-runner.jar",ActionOnFailure=TERMINATE_CLUSTER,Args=["spark-submit","--deploy-mode","cluster","--class","${job.name}","s3://${aws.bucket.name}/${jar.name}","s3://${aws.bucket.name}/${aws.model}","s3://${aws.bucket.name}/${aws.output}","s3://${aws.bucket.name}/${aws.test}"] \
		--log-uri s3://${aws.bucket.name}/${aws.log.dir} \
		--service-role EMR_DefaultRole \
		--ec2-attributes InstanceProfile=EMR_EC2_DefaultRole,SubnetId=${aws.subnet.id} \
		--region ${aws.region} \
		--enable-debugging \
		--auto-terminate

# Download output from S3.
download-output-aws: clean-local-output
	mkdir ${local.output}
	aws s3 sync s3://${aws.bucket.name}/${aws.output} ${local.output}
	
# Download model from S3.
download-model-aws: clean-local-model
	mkdir ${local.model}
	aws s3 sync s3://${aws.bucket.name}/${aws.model} ${local.model}

download-log-aws:
	mkdir ${local.log}
	aws s3 sync s3://${aws.bucket.name}/${aws.log} ${local.log}

# Package for release.
distro:
	rm SparkPageRank.tar.gz
	rm SparkPageRank.zip
	rm -rf build
	mkdir -p build/deliv/SparkPageRank/main/scala/pagerank
	cp -r src/main/scala/pagerank/* build/deliv/SparkPageRank/main/scala/pagerank
	cp pom.xml build/deliv/SparkPageRank
	cp Makefile build/deliv/SparkPageRank
	cp README.txt build/deliv/SparkPageRank
	tar -czf SparkPageRank.tar.gz -C build/deliv SparkPageRank
	cd build/deliv && zip -rq ../../SparkPageRank.zip SparkPageRank
