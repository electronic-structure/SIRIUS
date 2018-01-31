all: atoms dft-loop tests unit-tests

atoms:
	cd atoms && $(MAKE)
dft-loop:
	cd dft_loop && $(MAKE)
tests:
	cd tests && $(MAKE)
unit-tests:
	cd unit_tests && $(MAKE)

clean:
	cd atoms && $(MAKE) clean
	cd dft_loop && $(MAKE) clean
	cd tests && $(MAKE) clean
	cd unit_tests && $(MAKE) clean
