/*
 * util.h
 *
 *  Created on: 06/03/2015
 *      Author: pedroa
 */

#ifndef UTIL_H_
#define UTIL_H_
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <iostream>

namespace util{
//la funbcion de pedro esta mal :(
static int * vectorAleatoriosEnterosSinRepeticion(int minimo, int maximo, int cuantos){
        int total = maximo-minimo+1;
        int* numerosPorElegir = new int[total];
        int* numerosElegidos = new int[cuantos];
        // Inicializar la lista de elecciones posibles
        for(int i = 0; i < total; i++)
                numerosPorElegir[i] = minimo+i;

        for(int i=0; i < cuantos; i++)
        {
                int numeroElegido = rand() % (total-i);
                // Recordar el numero elegido
                numerosElegidos[i]=numerosPorElegir[numeroElegido];
                // Ponemos en numeros por elegir, el ultimo que era valido, asi
                // son todos validos hasta data.n-i-1
                numerosPorElegir[numeroElegido]=numerosPorElegir[total-i-1];
        }
        delete [] numerosPorElegir;
        return numerosElegidos;

};

static bool comprobarExistencia(int * vector, int tamagno, int numero){
	std::cout << tamagno << std::endl;
	for(int i=0; i < tamagno; i++){
		if(numero == vector[i]){
			return true;
		}
	}

	return false;
};

}


#endif /* UTIL_H_ */
