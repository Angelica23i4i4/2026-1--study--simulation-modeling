all:
	-@quarto render

release: all
	mkdir -p release
	cp -r _output/* release/

clean:
	-rm -rf _output

cleanall: clean
	-rm -rf .quarto release
