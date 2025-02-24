import argparse

def define_arguments_prompted(parser):
    parser.add_argument("--model_type", help="Type of the model to evaluate")
    parser.add_argument("--model_name", help="Name of the model to evaluate", default='', required=True)
    parser.add_argument("--setting", choices=['base', 'enhanced'], help="base or enhanced setting", required=True)
    parser.add_argument("--prompt_type", choices=['Defenced_InjecAgent','InjecAgent', 'hwchase17_react'], help="prompt type", required=True)
    parser.add_argument("--only_first_step", default=False, action='store_true', help="Only predict the first step")
    parser.add_argument("--use_cache", default=False, action='store_true', help="Use existing dataset")
    parser.add_argument("--only_get_score", default=False, action='store_true', help="Only get score without inference")
    parser.add_argument("--test_cases", default='dh', choices=['dh','ds'], help="Test mode (on which test cases to test)")
    parser.add_argument("--padding_tool", default='', choices=['CalculatorCalculate', 'TextProcessorCountCharacters', 'CurrencyConverterConvertCurrency', 
 'TimerToolkitSetTimer', 'WeatherToolkitGetCurrentWeather', 'WeatherToolkitGetWeatherForecast', 'WorldClockToolkitGetCurrentTime', 'WorldClockToolkitGetTimeDifference',''],
   help="Add padding actions to the requested actions")
    parser.add_argument("--test_non_existing_tool", default=False, action='store_true', help="Test the model with non-existing tool")
    parser.add_argument("--thought_injection", default='', choices=['harmless','harmful',''], help="Inject thoughts to the agent - harmless, harmful or none")
    parser.add_argument("--padding_tool_is_opp", default=False, action='store_true', help="Add padding actions to the requested actions in opposite order")
    parser.add_argument("--dist_loc", default='sL_sA',choices=['sL_sA','sL_eA','eL_sA','eL_eA'] ,help="Distractor location")
    parser.add_argument("--debug", default=False, action='store_true', help="Test the code and setup on a subset of examples (20)")

def parse_arguments():
    parser = argparse.ArgumentParser()
    define_arguments_prompted(parser)
    args = parser.parse_args()
    params = args.__dict__
    return params
