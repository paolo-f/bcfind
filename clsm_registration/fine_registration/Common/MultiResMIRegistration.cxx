/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    MultiResMIRegistration.cxx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) 2002 Insight Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/

#include <fstream>

#include "SimpleApp.h"
#include "itkExceptionObject.h"


int main(int argc, char *argv[])
{
  if ( argc < 6 or argc > 7) 
    {
    std::cout << "Parameter  file name missing ops" << std::endl;
    std::cout << std::endl;
    std::cout << "Usage: MultiResMIRegistration FixedImageName  MovingImageName  OutputImageName  ParametersFile FinalTransformFile [RegistrationLogFileName]" << std::endl;
    return 1;
    }

  // run the registration
  try
    {
    typedef itk::SimpleApp<signed short> AppType;
    AppType::Pointer theApp = AppType::New();
    
    theApp->SetFixedImageName( argv[1] ); 
    theApp->SetMovingImageName( argv[2] );
    theApp->SetOutputImageName( argv[3] );
    theApp->SetParameterFileName( argv[4] );
    theApp->SetFinalTransformFile( argv[5] );

    std::string path_filename(argv[1]);
    std::string substack_name;
    int found;
    while (substack_name.empty()){
      found = path_filename.find_last_of("/\\");
      substack_name = path_filename.substr(found+1);
      path_filename = path_filename.substr(0,found);
    }

    theApp->SetNameSubstack(substack_name);



    if (argc==7){
    	theApp->SetRegistrationLogFileName(argv[6]);
    }
    else if (argc==6){
	theApp->SetRegistrationLogFileName("");
    }


    
    theApp->Execute();

    }
  catch( itk::ExceptionObject& err)
    {
    std::cout << "Caught an ITK exception: " << std::endl;
    std::cout << err << std::endl;
    }
  catch(...)
    {
    std::cout << "Caught an non-ITK exception " << std::endl;
    }

  return 0;

}

