/*
 * util.h
 *
 *  Created on: 06/03/2015
 *      Author: pedroa
 */

#ifndef UTIL_H_
#define UTIL_H_

namespace util{
int * vectorAleatoriosEnterosSinRepeticion(int minimo, int maximo, int cuantos);
int argmax(double * vector, int tamagno);
bool comprobarExistencia(int * vector, int tamagno, int numero);

}


#endif /* UTIL_H_ */
