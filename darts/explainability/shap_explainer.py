"""
Shap wrapper Class
------------------------------

TODO
"""

from .explainability import ForecastingModelExplainer
from darts.models.forecasting.forecasting_model import ForecastingModel
from .. import models
from typing import Optional, Tuple, Union, Any, Callable, Dict, List, Sequence
from itertools import product
from abc import ABC, abstractmethod
from inspect import signature
import numpy as np
import pandas as pd
import shap
import torch

from ..utils import _build_tqdm_iterator
from ..timeseries import TimeSeries
from darts.logging import get_logger, raise_log, raise_if_not, raise_if

logger = get_logger(__name__)

class ShapExplainer(ForecastingModelExplainer):

    def __init__(self,
                model: ForecastingModel, 
                n: int,
                past_steps_explained: int,
                background_series: Optional[TimeSeries] = None
                ):

        """ Shap-based ForecastingModelExplainer

        This class is meant to wrap a shap explainability (https://github.com/slundberg/shap)
        specifically for time series.

        The idea is to get the shap values for each past time step of a given sample for a given
        timeserie prediction. past_steps_explained gives the past horizon we want to explain.

        TODO For now only univariate time series has been implemented.

        Warning

        This is only a shap value of direct influence and doesn't take into account relationships 
        between past values. Hence a past timestep could also have an indirect influence via the 
        intermediate timesteps.

        Parameters
        ----------
        model
            A forecasting model we want to explain.
        n
            number of predictions ahead we want to explain
        past_steps_explain
            A number of timesteps in the past of which we want to estimate the shap values. If x_t is 
            our prediction, past_steps_explain = p will take x_(t-1) ... x_(t-p) features in the past. 
            We will have as a result a matrix of n*past_steps_explain as there are n predictions to explain.
            This number is often linked to the model (order of the AR, input_chunk_length for a torch
            model etc...). For now we keep it generic.
        background_series
            the series we want to use to compare the foreground we want to explain. This is optional,
            for 2 reasons:
                * In general we want to keep the training_series of the model and this is the default one,
                but in case of multiple time series training (global or meta learning) we don't save them. 
                In this case we have to input a background_series.
                * We might want to compare to a reduced background distribution for computing-time sake.
        """
        super().__init__(model, past_steps_explained)

        if background_series is None:
            raise_if(
                self.model.training_series is None, 
                "A background time series has to be provided after fitting on multiple series, as"
                "no training series has been saved by the model.")
            background_series = self.model.training_series

        self.background_series = background_series
        # for the time serie, the expected return of the model is the mean of the time series itself
        self.expected_value = self.background_series.mean()

        if past_steps_explained > len(self.background_series)+1:
            raise_log(
                ValueError('The length of the timeseries must be at least past_step_explained+1'),
                logger
                )

        # Generate the Input dataset which will be used as a base distribution (background) to be
        # compared to the sample we want to explain (foreground). Format of shap input.
        self.X = create_shap_X(slicing(background_series, self.past_steps_explained+1))
        
        self.n = n

        self.shap_type = 'Kernel'

        # Test type of model - to choose the right shap method
        if isinstance(self.model, models.forecasting.torch_forecasting_model.TorchForecastingModel):
            self.shap_type = 'Deep'
        
        logger.info(('Shap {}Explainer will be used'.format(self.shap_type)))


        # The explainers is a list of explainers for each of the n predictions we are interested in.
        self.explainers = [self.create_explainer(i, self.X) for i in range(n)]

    def create_explainer(self,n: int, X) -> shap.KernelExplainer:
        if self.shap_type == 'Kernel':
            return shap.KernelExplainer(self.predict_wrapper_kernel_shap(n), self.X)
        elif self.shap_type == 'Deep':
            X_ = []
            X_np = self.X.values 
            for i in range(X_np.shape[0]):
                ts = X_np[i,:]
                X_.append(torch.tensor(ts.reshape(1, ts.shape[0], 1)))
            
            X = torch.Tensor(X_np.shape[0], ts.shape[0], 1)
            torch.cat(X_, out=X)

        return shap.DeepExplainer(self.predict_wrapper_deep_shap(n), X)

    def explain_at_timestamp(self, 
                          timestamp: Union[pd.Timestamp, int],
                          display: bool = False,
                          **kwargs) -> List:
        """
        Return a list of shap values arrays, one for each prediction from starting point timestamp input.

        Example:
        n=2, past_steps_explained=3, timestamp = 5 (can be also a datatime64)
        it will return a list of 2 elements, for timestamp 5 and timestamp 6, and each elements will have 
        3 shap values for each past timestep. 

        """

        # if timestamp.freq != self.background_series.freq:
        #     timestamp.freq = self.background_series.freq
        if not timestamp in self.background_series.time_index:
            raise_log(
                ValueError('The timestamp has not been found.'),
                logger
                )

        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)

        # if isinstance(timestamp, int):
        #     increments = [i for i in range(self.n)]
        # else:
        #     print(self.background_series.freq)
        #     increments = [pd.Timedelta(value=i, unit=self.background_series.freq.freqstr) for i in range(self.n)]
        #     print(increments)

        shap_values = [self.explainers[i].shap_values(self.X.loc[timestamp], **kwargs) for i in range(self.n)]

        if display:
            self.display_timestamp_shap_values(shap_values, timestamp)

        return shap_values

    def explain_from_input(self, 
                    foreground_series: TimeSeries,
                    display:bool = False,
                    **kwargs) -> List:
        """
        Return a list of shap values arrays, one for each prediction given the series input which constitutes
        the foreground sample here.

        Example:
        n=2, past_steps_explained=3, series of 7 elements
        it will return a list of 2 elements corresponding to the 2 predictions from the foreground sample 
        series[-past_steps_explained:] (so the last 3 elements), and each elements will have 3 shap values for each past timestep. 
        """

        foreground_series = pd.DataFrame(foreground_series.values().reshape(1, -1)).loc[0]
        shap_values = [
            self.explainers[i].shap_values(
                foreground_series, 
                **kwargs
                )
                for i in _build_tqdm_iterator(range(self.n), False)
            ]
        
        if display:
            self.display_input_shap_values(shap_values, foreground_series)

        return shap_values

    def display_timestamp_shap_values(self, shap_values, timestamp):
        
        shap.initjs()

        if isinstance(timestamp, str):
            timestamp = pd.Timestamp(timestamp)

        for i in range(len(self.explainers)):
            if isinstance(timestamp, pd.Timestamp):
                timestamp_i = timestamp+pd.Timedelta(value=i, unit=self.background_series.freq.freqstr)
            else:
                timestamp_i = timestamp+i
            df_tmp = self.X.rename(columns={c: "x_" + str(-self.past_steps_explained+j-1)  for j, c in enumerate(self.X.columns)})
            print(f'Current timestamp Shown x_{i} : {timestamp_i}')
            display(shap.force_plot(base_value = self.expected_value[0],
                shap_values = shap_values[i],
                features = df_tmp.loc[timestamp_i]
                )
            )

    def display_input_shap_values(self, shap_values, foreground_series):
        
        shap.initjs()
        for i in range(len(self.explainers)):
            print(f'Current timestamp Shown: x_t+{i}')
            display(shap.force_plot(base_value = self.expected_value[0],
                shap_values = shap_values[i],
                features = foreground_series
                )
            )

    # TODO for multivariate and or covariates, one will have to map correctly
    def predict_wrapper_kernel_shap(self, idx):
        """
        Creates a wraper for the right function to be used to predict in shap algorithm

        Parameters
        ----------
        idx
        index representing which prediction we compute, (between 0 and n).

        """
        def _f(X: np.ndarray) -> np.ndarray:
            o = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                o[i] = self.model.predict(
                                    self.n,
                                    series = TimeSeries.from_values(X[i,:]),
                                    num_samples=1
                                    ).values()[idx][0]
            return o
        return _f
    
    def predict_wrapper_deep_shap(self, idx):
        """
        Creates a wraper for the right function to be used to predict in shap algorithm

        Parameters
        ----------
        idx
        index representing which prediction we compute, (between 0 and n).

        """
        device = self.model.device
        return self.model.model.to(device)




def slicing(ts, slice_length) -> List:
    """Creates a list of sliced timeseries of length slice_length
    
    Parameters
    ----------
    ts
        An univariate timeseries
    slice_length
        the length of the sliced series

    """
    list_slices = []
    for idx in range(len(ts.time_index)-slice_length):
        list_slices.append(ts.slice(start_ts=ts.time_index[idx], end_ts=ts.time_index[idx+slice_length]))
    return list_slices

def create_shap_X(slices) -> pd.DataFrame:
    """ Creates the input shap needs from the time series. Can be numpy array or dataframe,
    For now we chose dataframe, as we can keep track of the time index.

    Parameters
    ----------
    A list of timeseries of length past_steps_explained+1
    """
    X = pd.DataFrame()
    
    for sl in slices:
        X = X.append(pd.DataFrame(sl.values().reshape(1, -1), index=[sl.time_index[-1]]))
    
    return X.drop(X.columns[-1], axis=1)


