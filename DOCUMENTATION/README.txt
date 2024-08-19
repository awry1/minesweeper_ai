README.txt - Co w folderach piszczy?

1) W folderze analytical przechowywane są datasety (na których uczone były/będą modele) oraz logi z win ratio. Datasety mają mieć w miarę możliwości 1000 iteracji (cries in train CNN).

2) W folderze neural networks są przechowywane "wyuczone" modele (300 epochów, takie same lr, gammy, itp.), logi z trenowania i testowania oraz drzewa decyzyjne (ofc tylko 5x5). Wszystko uporządkowane jest w odpowiednie foldery. 
WAŻNE: ilość iteracji w nazwie folderu dotyczy rozmiaru datasetu użytego do trenowania, nie ilości iteracji programu solve.py

UWAGI:
1) Pamiętajcie że rozmiar planszy musi być taki sam przy trenowaniu i testowaniu.

2) Gdzie się da to seed 'alamakota' żeby były jak najbardziej zbliżone wyniki
Chyba że testowanie to wypadałoby przetestować na jeszce innym datasecie niż treningowy, proponuję seed 'kotmaale'
