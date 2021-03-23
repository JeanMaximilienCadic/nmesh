docker rmi -f jcadic/nmesh
docker build . -t jcadic/nmesh
docker run -it --rm jcadic/nmesh bash
