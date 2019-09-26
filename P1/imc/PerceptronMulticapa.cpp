/*********************************************************************
* File : PerceptronMulticapa.cpp
* Date : 2018
*********************************************************************/

#include "PerceptronMulticapa.h"
#include "util.h"


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <limits>
#include <math.h>


using namespace imc;
using namespace std;
using namespace util;

// ------------------------------
// CONSTRUCTOR: Dar valor por defecto a todos los parámetros
PerceptronMulticapa::PerceptronMulticapa(){
	this->nNumCapas = 2;
	this->pCapas = NULL;
	this->dDecremento = 1;
	this->dValidacion = 0.2;
	this->dEta = 0.1;
	this->dMu = 0.9;
}

// ------------------------------
// Reservar memoria para las estructuras de datos

void PerceptronMulticapa::inicializar(int nl, int npl[]) {
	this->nNumCapas = nl;
	this->pCapas = new Capa[this->nNumCapas];


	for(int i=0; i < this->nNumCapas; i++){
		this->pCapas[i].nNumNeuronas = npl[i];
		this->pCapas[i].pNeuronas = new Neurona[this->pCapas[i].nNumNeuronas];

		if(i == 0){
			for(int j=0; j < this->pCapas[0].nNumNeuronas; j++){
				this->pCapas[0].pNeuronas[j].deltaW = NULL;
				this->pCapas[0].pNeuronas[j].ultimoDeltaW = NULL;
				this->pCapas[0].pNeuronas[j].w = NULL;
				this->pCapas[0].pNeuronas[j].wCopia = NULL;
			}
			continue;
		}

		for(int j=0; j < this->pCapas[i].nNumNeuronas; j++){
			this->pCapas[i].pNeuronas[j].deltaW = new double[this->pCapas[i-1].nNumNeuronas + 1];
			this->pCapas[i].pNeuronas[j].ultimoDeltaW = new double[this->pCapas[i-1].nNumNeuronas + 1];

			for(int k=0; k<this->pCapas[i-1].nNumNeuronas + 1; k++){
				this->pCapas[i].pNeuronas[j].ultimoDeltaW[k] = 0.0;
			}

			this->pCapas[i].pNeuronas[j].w = new double[this->pCapas[i-1].nNumNeuronas + 1];
			this->pCapas[i].pNeuronas[j].wCopia = new double[this->pCapas[i-1].nNumNeuronas + 1];
		}
	}
}


// ------------------------------
// DESTRUCTOR: liberar memoria
PerceptronMulticapa::~PerceptronMulticapa() {
	liberarMemoria();
}


// ------------------------------
// Liberar memoria para las estructuras de datos
void PerceptronMulticapa::liberarMemoria() {
	for(int i=0; i < this->nNumCapas; i++){
		if(i == 0){
			for(int j=0; j < this->pCapas[0].nNumNeuronas; j++){
				delete this->pCapas[0].pNeuronas;
			}
		}
		else{
			for(int j=0; j < this->pCapas[i].nNumNeuronas; j++){
				delete this->pCapas[i].pNeuronas[j].deltaW;
				delete this->pCapas[i].pNeuronas[j].ultimoDeltaW;
				delete this->pCapas[i].pNeuronas[j].w;
				delete this->pCapas[i].pNeuronas[j].wCopia;
			}

			delete this->pCapas[i].pNeuronas;
		}
	}

	delete this->pCapas;
}

// ------------------------------
// Rellenar todos los pesos (w) aleatoriamente entre -1 y 1
void PerceptronMulticapa::pesosAleatorios() {
	for(int i=1; i < this->nNumCapas; i++){
		std::cout << this->pCapas[i].nNumNeuronas << std::endl;
		for(int j = 0; j < this->pCapas[i].nNumNeuronas; j++){
			for(int k = 0; k < this->pCapas[i-1].nNumNeuronas + 1; k++){
				this->pCapas[i].pNeuronas[j].w[k] =
								(rand()/(double)RAND_MAX)*(1-(-1)) + (-1);
			}
		}
	}
}

// ------------------------------
// Alimentar las neuronas de entrada de la red con un patrón pasado como argumento
void PerceptronMulticapa::alimentarEntradas(double* input) {
	for(int i=0; i < this->pCapas[0].nNumNeuronas; i++){
		this->pCapas[0].pNeuronas[i].x = input[i];
	}
}

// ------------------------------
// Recoger los valores predichos por la red (out de la capa de salida) y almacenarlos en el vector pasado como argumento
void PerceptronMulticapa::recogerSalidas(double* output)
{
	for(int i=0; i < this->pCapas[this->nNumCapas-1].nNumNeuronas; i++){
		output[i] = this->pCapas[this->nNumCapas-1].pNeuronas[i].x;
	}
}

// ------------------------------
// Hacer una copia de todos los pesos (copiar w en copiaW)
void PerceptronMulticapa::copiarPesos() {
	for(int i=1; i < this->nNumCapas; i++){
		for(int j = 0; j < this->pCapas[i].nNumNeuronas; j++){
			for(int k = 0; k < this->pCapas[i-1].nNumNeuronas + 1; k++){
				this->pCapas[i].pNeuronas[j].wCopia[k] = this->pCapas[i].pNeuronas[j].w[k];
			}
		}
	}
}

// ------------------------------
// Restaurar una copia de todos los pesos (copiar copiaW en w)
void PerceptronMulticapa::restaurarPesos() {
	for(int i=1; i < this->nNumCapas; i++){
		for(int j = 0; j < this->pCapas[i].nNumNeuronas; j++){
			for(int k = 0; k < this->pCapas[i-1].nNumNeuronas + 1; k++){
				this->pCapas[i].pNeuronas[j].w[k] = this->pCapas[i].pNeuronas[j].wCopia[k];
			}
		}
	}
}

// ------------------------------
// Calcular y propagar las salidas de las neuronas, desde la primera capa hasta la última
void PerceptronMulticapa::propagarEntradas() {
	double net;
	for(int i=1; i < this->nNumCapas; i++){
		for(int j=0; j < this->pCapas[i].nNumNeuronas; j++){
			net = 0.0;
			for(int k=1; k < this->pCapas[i-1].nNumNeuronas + 1; k++){
				net += this->pCapas[i].pNeuronas[j].w[k] * this->pCapas[i-1].pNeuronas[k-1].x;
			}

			net += this->pCapas[i].pNeuronas[j].w[0];
			this->pCapas[i].pNeuronas[j].x = 1.0 / (1 + exp(-net));
		}
	}
}

// ------------------------------
// Calcular el cerror de salida (MSE) del out de la capa de salida con respecto a un vector objetivo y devolverlo
double PerceptronMulticapa::calcularErrorSalida(double* target) {
	double mse = 0.0;
	for(int i=0; i < this->pCapas[this->nNumCapas - 1].nNumNeuronas; i++){
		mse += pow(target[i] - this->pCapas[this->nNumCapas - 1].pNeuronas[i].x, 2);
	}

	return mse / (double)this->pCapas[this->nNumCapas - 1].nNumNeuronas;
}


// ------------------------------
// Retropropagar el error de salida con respecto a un vector pasado como argumento, desde la última capa hasta la primera
void PerceptronMulticapa::retropropagarError(double* objetivo) {
	for(int i=0; i < this->pCapas[this->nNumCapas - 1].nNumNeuronas; i++){
		double out = this->pCapas[this->nNumCapas - 1].pNeuronas[i].x;
		this->pCapas[this->nNumCapas - 1].pNeuronas[i].dX =
				- (objetivo[i] - out)*out*(1-out);
	}

	for(int j=this->nNumCapas - 2; j >= 1; j--){
		for(int k=0; k < this->pCapas[j].nNumNeuronas; k++){
			double aux = 0.0, out = this->pCapas[j].pNeuronas[k].x;
			for(int l=0; l < this->pCapas[j+1].nNumNeuronas; l++){
				aux += this->pCapas[j+1].pNeuronas[l].w[k+1]*this->pCapas[j+1].pNeuronas[l].dX;
			}

			this->pCapas[j].pNeuronas[k].dX = aux * out * (1-out);
		}
	}
}

// ------------------------------
// Acumular los cambios producidos por un patrón en deltaW
void PerceptronMulticapa::acumularCambio() {
	for(int i=1; i < this->nNumCapas; i++){
		for(int j=0; j < this->pCapas[i].nNumNeuronas; j++){
			for(int k=1; k < this->pCapas[i-1].nNumNeuronas + 1; k++){
				this->pCapas[i].pNeuronas[j].deltaW[k] +=
						this->pCapas[i].pNeuronas[j].dX * this->pCapas[i-1].pNeuronas[k-1].x;
			}

			this->pCapas[i].pNeuronas[j].deltaW[0] += this->pCapas[i].pNeuronas[j].dX;
		}
	}
}

// ------------------------------
// Actualizar los pesos de la red, desde la primera capa hasta la última
void PerceptronMulticapa::ajustarPesos() {
	double modifiedEta = 0.0;

	for(int i=1; i < this->nNumCapas; i++){
		modifiedEta = pow(this->dDecremento,-(this->nNumCapas - 1 - i))*this->dEta;
		for(int j=0; j < this->pCapas[i].nNumNeuronas; j++){
			for(int k=1; k < this->pCapas[i-1].nNumNeuronas + 1; k++){
				this->pCapas[i].pNeuronas[j].w[k] +=
						(- modifiedEta*this->pCapas[i].pNeuronas[j].deltaW[k])
						- this->dMu*(modifiedEta*this->pCapas[i].pNeuronas[j].ultimoDeltaW[k]);

				this->pCapas[i].pNeuronas[j].ultimoDeltaW[k] = this->pCapas[i].pNeuronas[j].deltaW[k];
			}

			this->pCapas[i].pNeuronas[j].w[0] +=
						(- modifiedEta*this->pCapas[i].pNeuronas[j].deltaW[0])
						- this->dMu*(modifiedEta*this->pCapas[i].pNeuronas[j].ultimoDeltaW[0]);

			this->pCapas[i].pNeuronas[j].ultimoDeltaW[0] = this->pCapas[i].pNeuronas[j].deltaW[0];
		}
	}
}

// ------------------------------
// Imprimir la red, es decir, todas las matrices de pesos
void PerceptronMulticapa::imprimirRed() {
	for(int i=1; i < this->nNumCapas; i++){
		std::cout << "Capa " << i << std::endl;
		std::cout << "------" << std::endl;

		for(int j=0; j < this->pCapas[i].nNumNeuronas; j++){
			for(int k=0; k < this->pCapas[i-1].nNumNeuronas + 1; k++){
				std::cout << this->pCapas[i].pNeuronas[j].w[k] << " ";
			}
			std::cout << std::endl;
		}
	}
}

// ------------------------------
// Simular la red: propagar las entradas hacia delante, retropropagar el error y ajustar los pesos
// entrada es el vector de entradas del patrón y objetivo es el vector de salidas deseadas del patrón
void PerceptronMulticapa::simularRedOnline(double* entrada, double* objetivo) {
	for(int i=1; i < this->nNumCapas; i++){
		for(int j=0; j < this->pCapas[i].nNumNeuronas; j++){
			for(int k=0; k < this->pCapas[i-1].nNumNeuronas + 1; k++){
				this->pCapas[i].pNeuronas[j].deltaW[k] = 0.0;
			}
		}
	}

	this->alimentarEntradas(entrada);
	this->propagarEntradas();
	this->retropropagarError(objetivo);
	this->acumularCambio();
	this->ajustarPesos();
}

// ------------------------------
// Leer una matriz de datos a partir de un nombre de fichero y devolverla
Datos* PerceptronMulticapa::leerDatos(const char *archivo) {
	ifstream f(archivo);

	if(!f.is_open()){
		std::cout << "Error" << std::endl;
		exit(-1);
	}

	Datos *datos = new Datos[1];

	f >> datos->nNumEntradas >> datos->nNumSalidas >> datos->nNumPatrones;

	datos->entradas = new double*[datos->nNumPatrones];
	datos->salidas = new double*[datos->nNumPatrones];

	for(int i=0; i < datos->nNumPatrones; i++){
		datos->entradas[i] = new double[datos->nNumEntradas];
		datos->salidas[i] = new double[datos->nNumSalidas];
	}

	for(int i=0; i < datos->nNumPatrones; i++){
		for(int j=0; j < datos->nNumEntradas; j++){
			f >> datos->entradas[i][j];
		}

		for(int k=0; k < datos->nNumSalidas; k++){
			f >> datos->salidas[i][k];
		}
	}

	return datos;
}

// ------------------------------
// Entrenar la red on-line para un determinado fichero de datos
void PerceptronMulticapa::entrenarOnline(Datos* pDatosTrain) {
	int i;
	for(i=0; i<pDatosTrain->nNumPatrones; i++){
		simularRedOnline(pDatosTrain->entradas[i], pDatosTrain->salidas[i]);
	}
}

// ------------------------------
// Probar la red con un conjunto de datos y devolver el error MSE cometido
double PerceptronMulticapa::test(Datos* pDatosTest) {
	double mse = 0.0;
	for(int i=0; i<pDatosTest->nNumPatrones; i++){
		this->alimentarEntradas(pDatosTest->entradas[i]);
		this->propagarEntradas();
		mse += this->calcularErrorSalida(pDatosTest->salidas[i]);
	}

	return mse / pDatosTest->nNumPatrones;
}

// OPCIONAL - KAGGLE
// Imprime las salidas predichas para un conjunto de datos.
// Utiliza el formato de Kaggle: dos columnas (Id y Predicted)
void PerceptronMulticapa::predecir(Datos* pDatosTest)
{
	int i;
	int j;
	int numSalidas = pCapas[nNumCapas-1].nNumNeuronas;
	double * salidas = new double[numSalidas];
	
	cout << "Id,Predicted" << endl;
	
	for (i=0; i<pDatosTest->nNumPatrones; i++){

		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		recogerSalidas(salidas);
		
		cout << i;

		for (j = 0; j < numSalidas; j++)
			cout << "," << salidas[j];
		cout << endl;

	}
}

// ------------------------------
// Ejecutar el algoritmo de entrenamiento durante un número de iteraciones, utilizando pDatosTrain
// Una vez terminado, probar como funciona la red en pDatosTest
// Tanto el error MSE de entrenamiento como el error MSE de test debe calcularse y almacenarse en errorTrain y errorTest
void PerceptronMulticapa::ejecutarAlgoritmoOnline(Datos * pDatosTrain, Datos * pDatosTest, int maxiter, double *errorTrain, double *errorTest)
{
	int countTrain = 0;

	// Inicialización de pesos
	pesosAleatorios();

	double minTrainError = 0;
	int numSinMejorar;
	double testError = 0;

	double validationError = 0;
	double lastValidationError = 0;
	int numSinMejorarValidacion = 0;
	Datos * pDatosValidacion = new Datos[1];
	Datos * pDatosTrain2 = new Datos[1];


	// Generar datos de validación
	if(dValidacion > 0 && dValidacion < 1){
		double numPatrones = pDatosTrain->nNumPatrones*this->dValidacion;
		int * indicePatronesValidacion = vectorAleatoriosEnterosSinRepeticion(0,
				pDatosTrain->nNumPatrones - 1, (int)numPatrones);

		pDatosValidacion->nNumPatrones = (int)numPatrones;
		pDatosValidacion->nNumEntradas = pDatosTrain->nNumEntradas;
		pDatosValidacion->nNumSalidas = pDatosTrain->nNumSalidas;
		pDatosValidacion->entradas = new double*[pDatosValidacion->nNumPatrones];
		pDatosValidacion->salidas = new double*[pDatosValidacion->nNumPatrones];

		for(int i=0; i < pDatosValidacion->nNumPatrones; i++){
			pDatosValidacion->entradas[i] = new double[pDatosTrain->nNumEntradas];
			pDatosValidacion->salidas[i] = new double[pDatosTrain->nNumSalidas];
		}

		pDatosTrain2->nNumPatrones = pDatosTrain->nNumPatrones - (int)numPatrones;
		pDatosTrain2->nNumEntradas = pDatosTrain->nNumEntradas;
		pDatosTrain2->nNumSalidas = pDatosTrain->nNumSalidas;
		pDatosTrain2->entradas = new double*[pDatosTrain2->nNumPatrones];
		pDatosTrain2->salidas = new double*[pDatosTrain2->nNumPatrones];

		for(int i=0; i < pDatosTrain2->nNumPatrones; i++){
			pDatosTrain2->entradas[i] = new double[pDatosTrain->nNumEntradas];
			pDatosTrain2->salidas[i] = new double[pDatosTrain->nNumSalidas];
		}

		for(int i=0; i < pDatosValidacion->nNumPatrones; i++){
			for(int j=0; j < pDatosValidacion->nNumEntradas; j++){
				pDatosValidacion->entradas[i][j] = pDatosTrain->entradas[indicePatronesValidacion[i]][j];
			}

			for(int k=0; k < pDatosValidacion->nNumSalidas; k++){
				pDatosValidacion->salidas[i][k] = pDatosTrain->salidas[indicePatronesValidacion[i]][k];
			}
		}


		int indice = 0;
		for(int j=0; j < pDatosTrain->nNumPatrones; j++){
			if(!comprobarExistencia(indicePatronesValidacion, (int)numPatrones, j)){
				for(int k=0; k < pDatosTrain2->nNumEntradas; k++){
					pDatosTrain2->entradas[indice][k] = pDatosTrain->entradas[j][k];
				}

				for(int l=0; l < pDatosTrain2->nNumSalidas; l++){
					pDatosTrain2->salidas[indice][l] = pDatosTrain->salidas[j][l];
				}
				indice++;
			}
		}

	}


	// Aprendizaje del algoritmo
	do {
		double trainError = 0.0;

		if(this->dValidacion > 0 && this->dValidacion < 1){
			entrenarOnline(pDatosTrain2);
			trainError = test(pDatosTrain2);
		}
		else{
			entrenarOnline(pDatosTrain);
			trainError = test(pDatosTrain);
		}

		if(countTrain==0 || trainError < minTrainError){
			minTrainError = trainError;
			copiarPesos();
			numSinMejorar = 0;
		}
		else if( (trainError-minTrainError) < 0.00001)
			numSinMejorar = 0;
		else
			numSinMejorar++;

		if(numSinMejorar==50){
			cout << "Salida porque no mejora el entrenamiento!!"<< endl;
			restaurarPesos();
			countTrain = maxiter;
		}

		if(dValidacion > 0 && dValidacion < 1){
			validationError = this->test(pDatosValidacion);
			if(countTrain == 0){
				lastValidationError = validationError;
			}
			else if ((validationError - lastValidationError) < 0.00001){
				numSinMejorarValidacion = 0;
			}
			else{
				numSinMejorarValidacion++;
			}

			lastValidationError = validationError;

			if(numSinMejorarValidacion == 50){
				cout << "Early Stopping" << endl;
				this->restaurarPesos();
				countTrain = maxiter;
			}
		}

		countTrain++;

		// Comprobar condiciones de parada de validación y forzar
		// OJO: en este caso debemos guardar el error de validación anterior, no el mínimo
		// Por lo demás, la forma en que se debe comprobar la condición de parada es similar
		// a la que se ha aplicado más arriba para el error de entrenamiento

		cout << "Iteración " << countTrain << "\t Error de entrenamiento: " << trainError << "\t Error de validación: " << validationError << endl;

	} while ( countTrain<maxiter );

	cout << "PESOS DE LA RED" << endl;
	cout << "===============" << endl;
	imprimirRed();

	cout << "Salida Esperada Vs Salida Obtenida (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<pDatosTest->nNumPatrones; i++){
		double* prediccion = new double[pDatosTest->nNumSalidas];

		// Cargamos las entradas y propagamos el valor
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		recogerSalidas(prediccion);
		for(int j=0; j<pDatosTest->nNumSalidas; j++)
			cout << pDatosTest->salidas[i][j] << " -- " << prediccion[j] << " ";
		cout << endl;
		delete[] prediccion;

	}

	testError = test(pDatosTest);
	*errorTest=testError;
	*errorTrain=minTrainError;

}

// OPCIONAL - KAGGLE
//Guardar los pesos del modelo en un fichero de texto.
bool PerceptronMulticapa::guardarPesos(const char * archivo)
{
	// Objeto de escritura de fichero
	ofstream f(archivo);

	if(!f.is_open())
		return false;

	// Escribir el numero de capas y el numero de neuronas en cada capa en la primera linea.
	f << nNumCapas;

	for(int i = 0; i < nNumCapas; i++)
		f << " " << pCapas[i].nNumNeuronas;
	f << endl;

	// Escribir los pesos de cada capa
	for(int i = 1; i < nNumCapas; i++)
		for(int j = 0; j < pCapas[i].nNumNeuronas; j++)
			for(int k = 0; k < pCapas[i-1].nNumNeuronas + 1; k++)
				f << pCapas[i].pNeuronas[j].w[k] << " ";

	f.close();

	return true;

}

// OPCIONAL - KAGGLE
//Cargar los pesos del modelo desde un fichero de texto.
bool PerceptronMulticapa::cargarPesos(const char * archivo)
{
	// Objeto de lectura de fichero
	ifstream f(archivo);

	if(!f.is_open())
		return false;

	// Número de capas y de neuronas por capa.
	int nl;
	int *npl;

	// Leer número de capas.
	f >> nl;

	npl = new int[nl];

	// Leer número de neuronas en cada capa.
	for(int i = 0; i < nl; i++)
		f >> npl[i];

	// Inicializar vectores y demás valores.
	inicializar(nl, npl);

	// Leer pesos.
	for(int i = 1; i < nNumCapas; i++)
		for(int j = 0; j < pCapas[i].nNumNeuronas; j++)
			for(int k = 0; k < pCapas[i-1].nNumNeuronas + 1; k++)
				f >> pCapas[i].pNeuronas[j].w[k];

	f.close();
	delete[] npl;

	return true;
}
