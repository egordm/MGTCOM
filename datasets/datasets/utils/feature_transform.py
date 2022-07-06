from collections import Counter
from typing import Optional, Union, List

import warnings
import itertools as it

import numpy as np
import pandas as pd

from feature_engine.encoding.base_encoder import BaseCategoricalTransformer


class MultiRareLabelEncoder(BaseCategoricalTransformer):
    def __init__(
            self,
            tol: float = 0.05,
            n_categories: int = 10,
            max_n_categories: Optional[int] = None,
            replace_with: Union[str, int, float] = "Rare",
            variables: Union[None, int, str, List[Union[str, int]]] = None,
            ignore_format: bool = False,
    ) -> None:

        if tol < 0 or tol > 1:
            raise ValueError("tol takes values between 0 and 1")

        if n_categories < 0 or not isinstance(n_categories, int):
            raise ValueError("n_categories takes only positive integer numbers")

        if max_n_categories is not None:
            if max_n_categories < 0 or not isinstance(max_n_categories, int):
                raise ValueError("max_n_categories takes only positive integer numbers")

        super().__init__(variables, ignore_format)
        self.tol = tol
        self.n_categories = n_categories
        self.max_n_categories = max_n_categories
        self.replace_with = replace_with
        self.ignore_format = True

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = self._check_fit_input_and_variables(X)

        self.encoder_dict_ = {}

        for var in self.variables_:
            counter = Counter(it.chain(*X[var].tolist()))

            if len(counter) > self.n_categories:

                # if the variable has more than the indicated number of categories
                # the encoder will learn the most frequent categories
                t = pd.Series(counter.values(), index=counter.keys())
                t = t / float(t.sum())

                # non-rare labels:
                freq_idx = t[t >= self.tol].index

                if self.max_n_categories:
                    self.encoder_dict_[var] = set(freq_idx[: self.max_n_categories])
                else:
                    self.encoder_dict_[var] = set(freq_idx)

            else:
                # if the total number of categories is smaller than the indicated
                # the encoder will consider all categories as frequent.
                warnings.warn(
                    "The number of unique categories for variable {} is less than that "
                    "indicated in n_categories. Thus, all categories will be "
                    "considered frequent".format(var)
                )
                self.encoder_dict_[var] = set(counter.keys())

        self._check_encoding_dictionary()

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._check_transform_input_and_state(X)

        for feature in self.variables_:
            X[feature] = X[feature].apply(lambda x: self.encoder_dict_[feature].intersection(x))

        return X

    def inverse_transform(self, X: pd.DataFrame):
        return self


class MOneHotEncoder(BaseCategoricalTransformer):
    def __init__(
            self,
            drop_last: bool = False,
            drop_last_binary: bool = False,
            variables: Union[None, int, str, List[Union[str, int]]] = None,
    ) -> None:

        if not isinstance(drop_last, bool):
            raise ValueError("drop_last takes only True or False")

        super().__init__(variables, True)
        self.drop_last = drop_last
        self.drop_last_binary = drop_last_binary

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X = self._check_fit_input_and_variables(X)

        self.encoder_dict_ = {}
        for var in self.variables_:
            self.encoder_dict_[var] = set(
                it.chain(*X[var].tolist())
            )

        self.n_features_in_ = X.shape[1]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = self._check_transform_input_and_state(X)

        for feature in self.variables_:
            for category in self.encoder_dict_[feature]:
                X[str(feature) + "_" + str(category)] = np.where(
                    X[feature].apply(lambda x: category in x), 1, 0
                )

        # drop the original non-encoded variables.
        X.drop(labels=self.variables_, axis=1, inplace=True)

        return X

    def inverse_transform(self, X: pd.DataFrame):
        return self
