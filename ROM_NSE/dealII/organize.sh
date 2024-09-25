echo "Download"
curl -X GET -u fischer:h05n04r09k.01 https://cloud.ifam.uni-hannover.de/remote.php/dav/files/fischer/Code/result/POD.tar.gz --output /home/ifam/fischer/Code/result/POD.tar.gz
curl -X GET -u fischer:h05n04r09k.01 https://cloud.ifam.uni-hannover.de/remote.php/dav/files/fischer/Code/result/ROM.tar.gz --output /home/ifam/fischer/Code/result/ROM.tar.gz

echo "copy"
cp /home/ifam/fischer/Code/result/POD.tar.gz /home/ifam/fischer/Code/result/POD_old.tar.gz
cp /home/ifam/fischer/Code/result/ROM.tar.gz /home/ifam/fischer/Code/result/ROM_old.tar.gz
 
echo "upload"
curl -X PUT -u fischer:h05n04r09k.01 https://cloud.ifam.uni-hannover.de/remote.php/dav/files/fischer/Code/result/POD.tar.gz -T /home/ifam/fischer/Code/result/POD.tar.gz
curl -X PUT -u fischer:h05n04r09k.01 https://cloud.ifam.uni-hannover.de/remote.php/dav/files/fischer/Code/result/ROM.tar.gz -T /home/ifam/fischer/Code/result/ROM.tar.gz
curl -X PUT -u fischer:h05n04r09k.01 https://cloud.ifam.uni-hannover.de/remote.php/dav/files/fischer/Code/result/POD.tar.gz -T /home/ifam/fischer/Code/result/POD_old.tar.gz
curl -X PUT -u fischer:h05n04r09k.01 https://cloud.ifam.uni-hannover.de/remote.php/dav/files/fischer/Code/result/ROM.tar.gz -T /home/ifam/fischer/Code/result/ROM_old.tar.gz
