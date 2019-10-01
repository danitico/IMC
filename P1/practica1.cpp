//============================================================================
// Introducción a los Modelos Computacionales
// Name        : practica1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // Para coger la hora time()
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <string.h>
#include <math.h>
#include "imc/PerceptronMulticapa.h"


using namespace imc;
using namespace std;
using namespace util;

int main(int argc, char **argv) {
    // Procesar los argumentos de la línea de comandos
    bool tflag = 0, Tflag = 0, iflag = 0, lflag = 0, hflag = 0, eflag = 0, mflag = 0;
    bool vflag = 0, dflag = 0, wflag = 0, pflag = 0, gflag = 0;
    char *Tvalue = NULL, *wvalue = NULL, *tvalue=NULL;
    int c, ivalue=0, lvalue=0, hvalue=0, dvalue=0;
    double evalue=0.0, mvalue=0.0, vvalue=0.0;

    opterr = 0;

    // a: opción que requiere un argumento
    // a:: el argumento requerido es opcional
    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:v:d:w:pg")) != -1)
    {
        // Se han añadido los parámetros necesarios para usar el modo opcional de predicción (kaggle).
        // Añadir el resto de parámetros que sean necesarios para la parte básica de las prácticas.
        switch(c){
        	case 't':
        		tflag = true;
        		tvalue = optarg;
        		break;

        	case 'T':
                Tflag = true;
                Tvalue = optarg;
                break;

            case 'i':
            	iflag = true;
            	ivalue = atoi(optarg);
            	break;

            case 'l':
            	lflag = true;
            	lvalue = atoi(optarg);
            	break;

            case 'h':
            	hflag = true;
            	hvalue = atoi(optarg);
            	break;

            case 'e':
            	eflag = true;
            	evalue = atof(optarg);
            	break;

            case 'm':
            	mflag = true;
            	mvalue = atof(optarg);
            	break;

            case 'v':
            	vflag = true;
            	vvalue = atof(optarg);
            	break;

            case 'd'://
            	dflag = true;
            	dvalue = atoi(optarg);
            	break;

            case 'w':
                wflag = true;
                wvalue = optarg;
                break;

            case 'p':
                pflag = true;
                break;

            case 'g':
            	gflag = true;
            	break;

            case '?':
                if (optopt == 't' || optopt == 'T' || optopt == 'i' || optopt == 'l'
                		|| optopt == 'h' || optopt == 'e' || optopt == 'm' || optopt == 'v'
                				|| optopt == 'd' || optopt == 'w')
                    fprintf (stderr, "La opción -%c requiere un argumento.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Opción desconocida `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Caracter de opción desconocido `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }

    if(!tflag && !pflag){
    	cout << "Se ha llamado mal al programa. Se necesita un fichero de entrenamiento" << endl;
    	exit(-1);
    }

    if (!pflag) {
        ////////////////////////////////////////
        // MODO DE ENTRENAMIENTO Y EVALUACIÓN //
        ///////////////////////////////////////

        // Objeto perceptrón multicapa
    	PerceptronMulticapa mlp;

        // Parámetros del mlp. Por ejemplo, mlp.dEta = valorQueSea;

        // Lectura de datos de entrenamiento y test: llamar a mlp.leerDatos(...)

        Datos *pDatosTest, *pDatosTrain;

        if(!Tflag){
            pDatosTrain = mlp.leerDatos(tvalue);
            pDatosTest = mlp.leerDatos(tvalue);
        }
        else{
        	pDatosTrain = mlp.leerDatos(tvalue);
        	pDatosTest = mlp.leerDatos(Tvalue);
        }

        if(!iflag){
        	ivalue = 1000;
        }

        if(!lflag){
        	lvalue = 1;
        }

        if(!hflag){
        	hvalue = 5;
        }

        if(!eflag){
        	evalue = 0.1;
        }

        if(!mflag){
        	mvalue = 0.9;
        }

        if(!vflag){
        	vvalue = 0.0;
        }

        if(!dflag){
        	dvalue = 1;
        }

        mlp.dEta = evalue;
        mlp.dMu = mvalue;
        mlp.dDecremento = dvalue;
        mlp.dValidacion = vvalue;


		int *topologia = new int[lvalue+2];
		topologia[0] = pDatosTrain->nNumEntradas;
		for(int i=1; i<(lvalue+2-1); i++){
			topologia[i] = hvalue;
		}
		topologia[lvalue+2-1] = pDatosTrain->nNumSalidas;

		mlp.inicializar(lvalue+2,topologia);




        // Semilla de los números aleatorios
        int semillas[] = {1,2,3,4,5};
        double *erroresTest = new double[5];
        double *erroresTrain = new double[5];
        double numPatrones=0.0;
        int * indicePatronesValidacion = NULL;
        double mejorErrorTest = 1.0, mejorErrorTrain = 1.0;
        int aux = 0;


        if(mlp.dValidacion > 0 && mlp.dValidacion < 1){
			numPatrones = pDatosTrain->nNumPatrones*mlp.dValidacion;

			if(numPatrones < 1){
				numPatrones = 1.0;
			}

			srand(time(NULL));
			indicePatronesValidacion = vectorAleatoriosEnterosSinRepeticion(0,
					pDatosTrain->nNumPatrones - 1, (int)numPatrones);
        }

        for(int i=0; i<5; i++){
        	cout << "**********" << endl;
        	cout << "SEMILLA " << semillas[i] << endl;
        	cout << "**********" << endl;
    		srand(semillas[i]);

    		mlp.ejecutarAlgoritmoOnline(pDatosTrain,pDatosTest,ivalue,&(erroresTrain[i]),&(erroresTest[i]),
    				indicePatronesValidacion, numPatrones, gflag, i);
    		cout << "Finalizamos => Error de test final: " << erroresTest[i] << endl;

    		if(gflag && erroresTrain[i] <= mejorErrorTrain){
    			mejorErrorTrain = erroresTrain[i];
    			aux=i;
    		}

            // (Opcional - Kaggle) Guardamos los pesos cada vez que encontremos un modelo mejor.
            if(wflag && erroresTest[i] <= mejorErrorTest)
            {
                mlp.guardarPesos(wvalue);
                mejorErrorTest = erroresTest[i];
            }
    	}

        cout << "HEMOS TERMINADO TODAS LAS SEMILLAS" << endl;

        double mediaErrorTest = 0, desviacionTipicaErrorTest = 0;
        double mediaErrorTrain = 0, desviacionTipicaErrorTrain = 0;
        
        // Calcular medias y desviaciones típicas de entrenamiento y test

        for(int i=0; i<5; i++){
        	mediaErrorTrain += erroresTrain[i];
        	cout << erroresTrain[i] << endl;
        	mediaErrorTest += erroresTest[i];
        }

        mediaErrorTrain /= 5;
        mediaErrorTest /= 5;

        for(int i=0; i<5; i++){
        	desviacionTipicaErrorTrain += pow(erroresTrain[i] - mediaErrorTrain, 2);
        	desviacionTipicaErrorTest += pow(erroresTest[i] - mediaErrorTest, 2);
        }

        desviacionTipicaErrorTrain = sqrt(desviacionTipicaErrorTrain / 5);
        desviacionTipicaErrorTest = sqrt(desviacionTipicaErrorTest / 5);

        cout << "INFORME FINAL" << endl;
        cout << "*************" << endl;
        cout << "Error de entrenamiento (Media +- DT): " << mediaErrorTrain << " +- " << desviacionTipicaErrorTrain << endl;
        cout << "Error de test (Media +- DT):          " << mediaErrorTest << " +- " << desviacionTipicaErrorTest << endl;

        if(gflag){
        	cout << "La mejor semilla es la " << aux << endl;
        }

        delete pDatosTest;
        delete pDatosTrain;
        delete topologia;
        delete erroresTest;
        delete erroresTrain;


        return EXIT_SUCCESS;
    }
    else {

        /////////////////////////////////
        // MODO DE PREDICCIÓN (KAGGLE) //
        ////////////////////////////////

        // Desde aquí hasta el final del fichero no es necesario modificar nada.
        
        // Objeto perceptrón multicapa
        PerceptronMulticapa mlp;

        // Inicializar red con vector de topología
        if(!wflag || !mlp.cargarPesos(wvalue))
        {
            cerr << "Error al cargar los pesos. No se puede continuar." << endl;
            exit(-1);
        }

        // Lectura de datos de entrenamiento y test: llamar a mlp.leerDatos(...)
        Datos *pDatosTest;
        pDatosTest = mlp.leerDatos(Tvalue);
        if(pDatosTest == NULL)
        {
            cerr << "El conjunto de datos de test no es válido. No se puede continuar." << endl;
            exit(-1);
        }

        mlp.predecir(pDatosTest);

        return EXIT_SUCCESS;
    }
}

