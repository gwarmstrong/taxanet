CXX = g++
CXXFLAGS = -Wall -Xpreprocessor -fopenmp -lomp -O3
PROGS = classify

.PHONY: all install clean

all: $(PROGS)

install: $(PROGS)
	cp $(PROGS) $(KRAKEN_DIR)/

clean:
	rm -f $(PROGS) *.o

db_shrink: krakendb.o quickfile.o

db_sort: krakendb.o quickfile.o

set_lcas: krakendb.o quickfile.o krakenutil.o seqreader.o

kmer_estimator: krakenutil.o seqreader.o

classify: krakendb.o quickfile.o krakenutil.o seqreader.o

# TODO this might not need all these files
hackdb: krakendb.o quickfile.o krakenutil.o seqreader.o

make_seqid_to_taxid_map: quickfile.o

krakenutil.o: krakenutil.cpp krakenutil.hpp
	$(CXX) $(CXXFLAGS) -c krakenutil.cpp

krakendb.o: krakendb.cpp krakendb.hpp quickfile.hpp
	$(CXX) $(CXXFLAGS) -c krakendb.cpp

seqreader.o: seqreader.cpp seqreader.hpp quickfile.hpp
	$(CXX) $(CXXFLAGS) -c seqreader.cpp

quickfile.o: quickfile.cpp quickfile.hpp
	$(CXX) $(CXXFLAGS) -c quickfile.cpp
