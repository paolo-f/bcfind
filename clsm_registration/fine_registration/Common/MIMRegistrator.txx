/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    MIMRegistrator.txx
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) 2002 Insight Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef _MIMRegistrator_txx
#define _MIMRegistrator_txx

#include "MIMRegistrator.h"


namespace itk
{

class CommandIterationUpdate : public itk::Command
{
private:
int counter;
int step;
std::ofstream RegistrationLogFileName;


public:
typedef CommandIterationUpdate Self;
typedef Command Superclass;
typedef SmartPointer<Self> Pointer;
itkNewMacro( Self );

~CommandIterationUpdate(){
RegistrationLogFileName.close();
}

protected:
CommandIterationUpdate():counter(0),step(100) {};

public:
void SetRegistrationLogFileName(std::string _RegistrationLogFileName){


 try{
 	RegistrationLogFileName.open(_RegistrationLogFileName.c_str(),std::ofstream::out);   
   }
  catch (std::ofstream::failure e) {
         std::cout << "Exception opening/writing file";
  }
  

}
typedef QuaternionRigidTransformGradientDescentOptimizerStoppingCriterion OptimizerType;
typedef const OptimizerType * OptimizerPointer;

void Execute(Object *caller, const EventObject & event)
{
Execute( (const Object *)caller, event);
}
void Execute(const Object * object, const EventObject & event)
{
OptimizerPointer optimizer =
dynamic_cast< OptimizerPointer >( object );
if( !(itk::IterationEvent().CheckEvent( &event )) )
{
return;
}

if ((int)optimizer->GetCurrentIteration() % step == 0){
	std::cout << optimizer->GetCurrentIteration() << " ";
	std::cout << optimizer->GetValue() << " ";
	std::cout << optimizer->GetCurrentPosition() << std::endl;
}

std::ostringstream ostr_iter; 
std::ostringstream ostr_value; 
ostr_iter << optimizer->GetCurrentIteration(); 
ostr_value << optimizer->GetValue(); 

RegistrationLogFileName << ostr_iter.str() <<","<<ostr_value.str()<<"\n";

RegistrationLogFileName.flush();

}
};




template <typename TFixedImage, typename TMovingImage>
MIMRegistrator<TFixedImage,TMovingImage>
::MIMRegistrator()
{
  // Images need to be set from the outside
  m_FixedImage  = NULL;
  m_MovingImage = NULL;

  // Set up internal registrator with default components
  m_Transform          = TransformType::New();
  m_Optimizer          = OptimizerType::New();
  m_Interpolator       = InterpolatorType::New();
  m_FixedImagePyramid  = FixedImagePyramidType::New();
  m_MovingImagePyramid = MovingImagePyramidType::New();
  m_Registration       = RegistrationType::New();

  m_Registration->SetTransform( m_Transform );
  m_Registration->SetOptimizer( m_Optimizer );

  
  //m_Registration->SetMetric( m_Metric );
  m_Registration->SetInterpolator( m_Interpolator );
  m_Registration->SetFixedImagePyramid( m_FixedImagePyramid );
  m_Registration->SetMovingImagePyramid( m_MovingImagePyramid );

  m_AffineTransform  = AffineTransformType::New();



   


  // Setup an registration observer
  typedef SimpleMemberCommand<Self> CommandType;
  typename CommandType::Pointer command = CommandType::New();
  command->SetCallbackFunction( this, &Self::StartNewLevel );

  m_Tag = m_Registration->AddObserver( IterationEvent(), command );

  // Default parameters
  m_NumberOfLevels = 1;
  m_TranslationScale = 1.0;
  m_MovingImageStandardDeviation = 0.4;
  m_FixedImageStandardDeviation = 0.4;
  m_NumberOfSpatialSamples = 50;  

  m_FixedImageShrinkFactors.Fill( 1 );
  m_MovingImageShrinkFactors.Fill( 1 );

  m_NumberOfIterations = UnsignedIntArray(1);
  m_NumberOfIterations.Fill( 10 );

  m_LearningRates = DoubleArray(1);
  m_LearningRates.Fill( 1e-4 );

  m_InitialParameters = ParametersType( m_Transform->GetNumberOfParameters() );
  m_InitialParameters.Fill( 0.0 );
  m_InitialParameters[3] = 1.0;
    
}


template <typename TFixedImage, typename TMovingImage>
MIMRegistrator<TFixedImage,TMovingImage>
::~MIMRegistrator()
{
  m_Registration->RemoveObserver( m_Tag );

}


template <typename TFixedImage, typename TMovingImage>
void
MIMRegistrator<TFixedImage,TMovingImage>
::Execute()
{
  


  //Setup an optimization observer
   typename CommandIterationUpdate::Pointer observer = CommandIterationUpdate::New(); 
   observer->SetRegistrationLogFileName(this->m_RegistrationLogFileName);
   m_Optimizer->AddObserver( IterationEvent(), observer );



  // Setup the optimizer
  typename OptimizerType::ScalesType scales( 
    m_Transform->GetNumberOfParameters() );

  scales.Fill( 1.0);
  
  for ( int j = 4; j < 7; j++ )
    {
    scales[j] = m_TranslationScale;
    }


  //select the metric
  if(m_MetricType==2){
      
  	m_NMattesMetric       = MattesNMIMetricType::New();
  	m_Registration->SetMetric( m_NMattesMetric );
        m_Optimizer->SetScales( scales );
	m_Optimizer->MinimizeOn();
	unsigned int numberOfBins = 50;
        //m_NMattesMetric->SetLogFile("/tmp/joint_pdf_ex");
        m_NMattesMetric->SetNumberOfHistogramBins( numberOfBins );
        m_NMattesMetric->SetNumberOfSpatialSamples( m_NumberOfSpatialSamples );
	
  }
  else if(m_MetricType==1){
      
  	m_MattesMetric       = MattesMIMetricType::New();
  	m_Registration->SetMetric( m_MattesMetric );
        m_Optimizer->SetScales( scales );
	m_Optimizer->MinimizeOn();
	unsigned int numberOfBins = 50;//50;//100;
        m_MattesMetric->SetNumberOfHistogramBins( numberOfBins );
        m_MattesMetric->SetNumberOfSpatialSamples( m_NumberOfSpatialSamples );
	
  }
  else if (m_MetricType==0){
	m_ViolaMetric	       = ViolaMIMetricType::New();
	m_Registration->SetMetric( m_ViolaMetric);
	m_Optimizer->SetScales( scales );
  	m_Optimizer->MaximizeOn();
	m_ViolaMetric->SetMovingImageStandardDeviation( m_MovingImageStandardDeviation );
  	m_ViolaMetric->SetFixedImageStandardDeviation( m_FixedImageStandardDeviation );
  	m_ViolaMetric->SetNumberOfSpatialSamples( m_NumberOfSpatialSamples );
  	
 
  }  

  // Setup the image pyramids
  m_FixedImagePyramid->SetNumberOfLevels( m_NumberOfLevels );
  m_FixedImagePyramid->SetStartingShrinkFactors( 
    m_FixedImageShrinkFactors.GetDataPointer() );

  m_MovingImagePyramid->SetNumberOfLevels( m_NumberOfLevels );
  m_MovingImagePyramid->SetStartingShrinkFactors(
    m_MovingImageShrinkFactors.GetDataPointer() );

  // Setup the registrator
  m_Registration->SetFixedImage( m_FixedImage );
  m_Registration->SetMovingImage( m_MovingImage );
  m_Registration->SetNumberOfLevels( m_NumberOfLevels );
 
  m_Registration->SetInitialTransformParameters( m_InitialParameters );

  m_Registration->SetFixedImageRegion( m_FixedImage->GetBufferedRegion() );

  try
    {
    m_Registration->Update();
    }
  catch( itk::ExceptionObject & err )
    {
    std::cout << "Caught an exception: " << std::endl;
    std::cout << err << std::endl;
    throw err;
    }

  
   double bestValue = m_Optimizer->GetValue();
   std::cout << " Best Metric value  = " << bestValue          << std::endl;

}


template <typename TFixedImage, typename TMovingImage>
const 
typename MIMRegistrator<TFixedImage,TMovingImage>
::ParametersType &
MIMRegistrator<TFixedImage,TMovingImage>
::GetTransformParameters()
{
  return m_Registration->GetLastTransformParameters();
}


template <typename TFixedImage, typename TMovingImage>
typename MIMRegistrator<TFixedImage,TMovingImage>
::AffineTransformPointer
MIMRegistrator<TFixedImage,TMovingImage>
::GetAffineTransform()
{
  m_Transform->SetParameters( m_Registration->GetLastTransformParameters() );
  
  m_AffineTransform->SetMatrix( m_Transform->GetMatrix() );
  m_AffineTransform->SetOffset( m_Transform->GetOffset() );

  return m_AffineTransform;
}



template <typename TFixedImage, typename TMovingImage>
void
MIMRegistrator<TFixedImage,TMovingImage>
::StartNewLevel()
{
  std::cout << "--- Starting level " << m_Registration->GetCurrentLevel()
            << std::endl;

  unsigned int level = m_Registration->GetCurrentLevel();
  if ( m_NumberOfIterations.Size() >= level + 1 )
    {
    m_Optimizer->SetNumberOfIterations( m_NumberOfIterations[level] );
    }

  if ( m_LearningRates.Size() >= level + 1 )
    {
    m_Optimizer->SetLearningRate( m_LearningRates[level] );
    }

  std::cout << " No. Iterations: " 
            << m_Optimizer->GetNumberOfIterations()
            << " Learning rate: "
            << m_Optimizer->GetLearningRate()
            << std::endl;

}


} // namespace itk


#endif
