main: test.c qbs.o
	gcc qbs.o test.c -Wall -g -lm -o testqbs

module: qbs.o cqbsmodule.c
	python setup.py build
	python setup.py install

qbs.o: qbs.c
	gcc -Wall -lm -c qbs.c

clean:
	rm *.o
	rm testqbs

install:
	python setup.py install