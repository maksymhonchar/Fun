# connect via ssh
ssh -i path_to_key.pem ubuntu@publicdns -p port_number

# copy files
scp -i path/to/key file/to/copy user@ec2-xx-xx-xxx-xxx.compute-1.amazonaws.com:path/to/file

# hadoop setup process

todo!

## java
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt-get update
sudo apt install openjdk-8-jdk-headless -y
(Strictly speaking, the JRE is enough to make Hadoop running. here: jdk for possible extras)

## others
sudo apt-get install build-essential
sudo apt-get install python

## hadoop
wget hadoop.tar.gz
tar -zxf hadoop.tar.gz
sudo mv hadoop /usr/local/hadoop


# todo

https://codethief.io/hadoop101/

https://dzone.com/articles/how-set-multi-node-hadoop

https://mfaizmzaki.com/2015/07/21/how-to-install-hadoop-multin-node-cluster-on-amazon-aws-ec2-instance/