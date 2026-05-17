JAVAC=javac
JAVA=java
SRC=$(wildcard *.java)
CLASSES=$(SRC:.java=.class)

all: compile

compile:
	$(JAVAC) $(SRC)

clean:
	rm -f *.class

jar: compile
	jar cfm search.jar manifest.txt *.class README.md

run-task1:
	$(JAVA) Task1Test

run-task2:
	$(JAVA) Task2Test