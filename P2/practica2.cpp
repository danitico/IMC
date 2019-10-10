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
#include <ctime>    // Para cojer la hora time()
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <string.h>
#include <math.h>
#include "imc/PerceptronMulticapa.h"
#include "imc/util.h"

using namespace imc;
using namespace std;

int main(int argc, char **argv) {
    // Procesar los argumentos de la línea de comandos
    bool tflag = 0, Tflag = 0, iflag = 0, lflag = 0, hflag = 0, eflag = 0, mflag = 0;
    bool vflag = 0, dflag = 0, oflag = 0, fflag = 0, wflag = 0, pflag = 0;
    bool sflag = 0;
    char *Tvalue = NULL, *wvalue = NULL, *tvalue=NULL;
    int c, ivalue=0, lvalue=0, hvalue=0, dvalue=0, fvalue = 0;
    double evalue=0.0, mvalue=0.0, vvalue=0.0;

    opterr = 0;

    // a: opción que requiere un argumento
    // a:: el argumento requerido es opcional
    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:v:d:of:sw:p")) != -1)
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

            case 'd':
            	dflag = true;
            	dvalue = atoi(optarg);
            	break;

            case 'o':
            	oflag = true;
            	break;

            case 'f':
            	fflag = true;
            	fvalue = atoi(optarg);
            	break;

            case 's':
            	sflag = true;
            	break;

            case 'w':
                wflag = true;
                wvalue = optarg;
                break;

            case 'p':
                pflag = true;
                break;

//            case 'g':
//            	gflag = true;
//            	break;

            case '?':
                if (optopt == 't' || optopt == 'T' || optopt == 'i' || optopt == 'l'
                		|| optopt == 'h' || optopt == 'e' || optopt == 'm' || optopt == 'v'
                				|| optopt == 'd' || optopt == 'f' || optopt == 'w')
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
        	hvalue = 4;
        }

        if(!eflag){
        	evalue = 0.7;
        }

        if(!mflag){
        	mvalue = 1;
        }

        if(!vflag){
        	vvalue = 0.0;
        }

        if(!dflag){
        	dvalue = 1;
        }

        if(!fflag){
        	fvalue = 0;
        }


        mlp.dEta = evalue;
        mlp.dMu = mvalue;
        mlp.dDecremento = dvalue;
        mlp.dValidacion = vvalue;
        mlp.bOnline = oflag;


    	// Inicializar vector topología
    	int *topologia = new int[lvalue+2];
    	int *tipo = new int[lvalue+2];

    	topologia[0] = pDatosTrain->nNumEntradas;
    	tipo[0] = 0;

    	for(int i=1; i<(lvalue+2-1); i++){
    		topologia[i] = hvalue;
    		tipo[i] = 0;
    	}
    	topologia[lvalue+2-1] = pDatosTrain->nNumSalidas;
    	tipo[lvalue+2-1] = (int)sflag;
    	mlp.inicializar(lvalue+2,topologia, tipo);


        // Semilla de los números aleatorios
        int semillas[] = {1,2,3,4,5};
        double *errores = new double[5];
        double *erroresTrain = new double[5];
        double *ccrs = new double[5];
        double *ccrsTrain = new double[5];
        double mejorErrorTest = 1.0, numPatrones = 0.0;
        int * indicePatronesValidacion = NULL;


        if(mlp.dValidacion > 0 && mlp.dValidacion < 1){
			numPatrones = pDatosTrain->nNumPatrones*mlp.dValidacion;

			if(numPatrones < 1){
				numPatrones = 1.0;
			}

			srand(time(NULL));
			indicePatronesValidacion = util::vectorAleatoriosEnterosSinRepeticion(0,
					pDatosTrain->nNumPatrones - 1, (int)numPatrones);
        }



        for(int i=0; i<5; i++){
        	cout << "**********" << endl;
        	cout << "SEMILLA " << semillas[i] << endl;
        	cout << "**********" << endl;
    		srand(semillas[i]);
    		mlp.ejecutarAlgoritmo(pDatosTrain,pDatosTest,ivalue,&(erroresTrain[i]),&(errores[i]),&(ccrsTrain[i]),&(ccrs[i]),fvalue, indicePatronesValidacion, numPatrones);
    		cout << "Finalizamos => CCR de test final: " << ccrs[i] << endl;

            // (Opcional - Kaggle) Guardamos los pesos cada vez que encontremos un modelo mejor.
            if(wflag && errores[i] <= mejorErrorTest)
            {
                mlp.guardarPesos(wvalue);
                mejorErrorTest = errores[i];
            }

        }


        double mediaError = 0, desviacionTipicaError = 0;
        double mediaErrorTrain = 0, desviacionTipicaErrorTrain = 0;
        double mediaCCR = 0, desviacionTipicaCCR = 0;
        double mediaCCRTrain = 0, desviacionTipicaCCRTrain = 0;

        for(int i=0; i<5; i++){
        	mediaErrorTrain += erroresTrain[i];
        	mediaError += errores[i];
        	mediaCCR += ccrs[i];
        	mediaCCRTrain += ccrsTrain[i];
        }

        mediaErrorTrain /= 5;
        mediaError /= 5;
        mediaCCR /= 5;
        mediaCCRTrain /= 5;

        for(int i=0; i<5; i++){
        	desviacionTipicaErrorTrain += pow(erroresTrain[i] - mediaErrorTrain, 2);
        	desviacionTipicaError += pow(errores[i] - mediaError, 2);
        	desviacionTipicaCCRTrain += pow(ccrsTrain[i] - mediaCCRTrain, 2);
        	desviacionTipicaCCR += pow(ccrs[i] - mediaCCR, 2);
        }

        desviacionTipicaErrorTrain = sqrt(desviacionTipicaErrorTrain / 5);
        desviacionTipicaError = sqrt(desviacionTipicaError / 5);
        desviacionTipicaCCRTrain = sqrt(desviacionTipicaCCRTrain / 5);
        desviacionTipicaCCR = sqrt(desviacionTipicaCCR / 5);

        cout << "HEMOS TERMINADO TODAS LAS SEMILLAS" << endl;

    	cout << "INFORME FINAL" << endl;
    	cout << "*************" << endl;
        cout << "Error de entrenamiento (Media +- DT): " << mediaErrorTrain << " +- " << desviacionTipicaErrorTrain << endl;
        cout << "Error de test (Media +- DT): " << mediaError << " +- " << desviacionTipicaError << endl;
        cout << "CCR de entrenamiento (Media +- DT): " << mediaCCRTrain << " +- " << desviacionTipicaCCRTrain << endl;
        cout << "CCR de test (Media +- DT): " << mediaCCR << " +- " << desviacionTipicaCCR << endl;

    	return EXIT_SUCCESS;
    } else {

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

