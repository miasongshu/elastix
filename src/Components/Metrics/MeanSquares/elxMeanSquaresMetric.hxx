#ifndef __elxMeanSquaresMetric_HXX__
#define __elxMeanSquaresMetric_HXX__

#include "elxMeanSquaresMetric.h"


namespace elastix
{
using namespace itk;

	/**
	 * ******************* Initialize ***********************
	 */

	template <class TElastix>
		void MeanSquaresMetric<TElastix>
		::Initialize(void) throw (ExceptionObject)
	{
		TimerPointer timer = TimerType::New();
		timer->StartTimer();
		this->Superclass1::Initialize();
		timer->StopTimer();
		elxout << "Initialization of MeanSquares metric took: "
			<< static_cast<long>( timer->GetElapsedClockSec() * 1000 ) << " ms." << std::endl;

	} // end Initialize


} // end namespace elastix


#endif // end #ifndef __elxMeanSquaresMetric_HXX__
