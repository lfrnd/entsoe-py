import logging
import warnings
from typing import Dict, Literal, Optional, Union

import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup
from bs4.builder import XMLParsedAsHTMLWarning
from pandas.tseries.offsets import YearBegin, YearEnd

from entsoe.exceptions import InvalidBusinessParameterError, InvalidParameterError, InvalidPSRTypeError

from .decorators import day_limited, documents_limited, paginated, retry, year_limited
from .exceptions import NoMatchingDataError, PaginationError
from .mappings import NEIGHBOURS, Area, lookup_area
from .parsers import (
    parse_activated_balancing_energy_prices,
    parse_aggregated_bids,
    parse_contracted_reserve,
    parse_crossborder_flows,
    parse_generation,
    parse_generic,
    parse_imbalance_prices_zip,
    parse_imbalance_volumes_zip,
    parse_installed_capacity_per_plant,
    parse_loads,
    parse_netpositions,
    parse_offshore_unavailability,
    parse_prices,
    parse_procured_balancing_capacity,
    parse_unavailabilities,
    parse_water_hydro,
)


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

__title__ = "entsoe-py"
__version__ = "0.6.16"
__author__ = "EnergieID.be, Frank Boerman"
__license__ = "MIT"

URL = "https://web-api.tp.entsoe.eu/api"


class EntsoeRawClient:
    # noinspection LongLine
    """
    Client to perform API calls and return the raw responses API-documentation:
    https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html#_request_methods

    Attributions: Parts of the code for parsing Entsoe responses were copied
    from https://github.com/tmrowco/electricitymap
    """

    def __init__(
        self,
        api_key: str,
        session: Optional[requests.Session] = None,
        retry_count: int = 1,
        retry_delay: int = 0,
        proxies: Optional[Dict] = None,
        timeout: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        api_key : str
        session : requests.Session
        retry_count : int
            number of times to retry the call if the connection fails
        retry_delay: int
            amount of seconds to wait between retries
        proxies : dict
            requests proxies
        timeout : int
        """
        if api_key is None:
            raise TypeError("API key cannot be None")
        self.api_key = api_key
        if session is None:
            session = requests.Session()
        self.session = session
        self.proxies = proxies
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.timeout = timeout

    @retry
    def _base_request(self, params: Dict, start: pd.Timestamp, end: pd.Timestamp) -> requests.Response:
        """
        Parameters
        ----------
        params : dict
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        requests.Response
        """
        start_str = self._datetime_to_str(start)
        end_str = self._datetime_to_str(end)

        base_params = {"securityToken": self.api_key, "periodStart": start_str, "periodEnd": end_str}
        params.update(base_params)

        logger.debug(f"Performing request to {URL} with params {params}")
        response = self.session.get(url=URL, params=params, proxies=self.proxies, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            soup = BeautifulSoup(response.text, "html.parser")
            text = soup.find_all("text")
            if len(text):
                error_text = soup.find("text").text
                if "No matching data found" in error_text:
                    raise NoMatchingDataError
                elif "check you request against dependency tables" in error_text:
                    raise InvalidBusinessParameterError
                elif "is not valid for this area" in error_text:
                    raise InvalidPSRTypeError
                elif "amount of requested data exceeds allowed limit" in error_text:
                    requested = error_text.split(" ")[-2]
                    allowed = error_text.split(" ")[-5]
                    raise PaginationError(
                        f"The API is limited to {allowed} elements per "
                        f"request. This query requested for {requested} "
                        f"documents and cannot be fulfilled as is."
                    )
                elif "requested data to be gathered via the offset parameter exceeds the allowed limit" in error_text:
                    requested = error_text.split(" ")[-9]
                    allowed = error_text.split(" ")[-30][:-2]
                    raise PaginationError(
                        f"The API is limited to {allowed} elements per "
                        f"request. This query requested for {requested} "
                        f"documents and cannot be fulfilled as is."
                    )
            raise e
        else:
            # ENTSO-E has changed their server to also respond with 200 if there is no data but all parameters are valid
            # this means we need to check the contents for this error even when status code 200 is returned
            # to prevent parsing the full response do a text matching instead of full parsing
            # also only do this when response type content is text and not for example a zip file
            if response.headers.get("content-type", "") == "application/xml":
                if "No matching data found" in response.text:
                    raise NoMatchingDataError
            return response

    @staticmethod
    def _datetime_to_str(dtm: pd.Timestamp) -> str:
        """
        Convert a datetime object to a string in UTC
        of the form YYYYMMDDhh00

        Parameters
        ----------
        dtm : pd.Timestamp
            Recommended to use a timezone-aware object!
            If timezone-naive, UTC is assumed

        Returns
        -------
        str
        """
        if dtm.tzinfo is not None and dtm.tzinfo != pytz.UTC:
            dtm = dtm.tz_convert("UTC")
        fmt = "%Y%m%d%H00"
        ret_str = dtm.round(freq="h").strftime(fmt)
        return ret_str

    def query_day_ahead_prices(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, offset: int = 0
    ) -> str:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        str
        """
        area = lookup_area(country_code)
        params = {"documentType": "A44", "in_Domain": area.code, "out_Domain": area.code, "offset": offset}
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_crossborder_balancing(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> str:
        area_from = lookup_area(country_code_from)
        area_to = lookup_area(country_code_to)
        params = {
            "documentType": "A88",
            "acquiring_Domain": area_from.code,
            "connecting_Domain": area_to.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_frr_rr_sharing(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> str:
        area_from = lookup_area(country_code_from)
        area_to = lookup_area(country_code_to)
        params = {
            "documentType": "A26",
            "processType": "A56",
            "businessType": "C22",
            "acquiring_Domain": area_from.code,
            "connecting_Domain": area_to.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_exchanged_reserve_capacity(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        offset: int = 0,
    ) -> str:
        area_from = lookup_area(country_code_from)
        area_to = lookup_area(country_code_to)
        params = {
            "documentType": "A26",
            "processType": "A46",
            "businessType": "C21",
            "acquiring_Domain": area_from.code,
            "connecting_Domain": area_to.code,
            "offset": offset,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_cross_zonal_balancing_capacity_allocation_and_use(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        type_marketagreement_type: str,
        process_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> str:
        area_from = lookup_area(country_code_from)
        area_to = lookup_area(country_code_to)
        params = {
            "documentType": "A38",
            "acquiring_Domain": area_from.code,
            "connecting_Domain": area_to.code,
            "processType": process_type,
            "Type_MarketAgreement.Type": type_marketagreement_type,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_netted_and_exchanged_volumes(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        process_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> str:
        """
        processType
        [M] A60 = mFRR with Scheduled Activation; A61 = mFRR with Direct Activation;
        A51 = Automatic Frequency Restoration Reserve; A63= Imbalance Netting
        """
        area_from = lookup_area(country_code_from)
        area_to = lookup_area(country_code_to)
        params = {
            "documentType": "B17",
            "processType": process_type,
            "acquiring_Domain": area_from.code,
            "connecting_Domain": area_to.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_balancing_energy_bids(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        process_type: str,
        offset: int = 0,
    ) -> str:
        """
        processType
        [M] A46 = Replacement reserve; A47 = Manual frequency restoration reserve;
        A51 = Automatic frequency restoration reserve

        Standard_MarketProduct
        [O] A01 = Standard; A05 = Standard mFRR scheduled activation; A07 = Standard mFRR direct activation

        Original_MarketProduct
        [O] A02 = Specific; A03 = Integrated Process; A04 = Local

        Direction
        [O] A01 = Up; A02 = Down
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A37",
            "businessType": "B74",
            "connecting_Domain": area.code,
            "processType": process_type,
            "offset": offset,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_elastic_demands(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        offset: int = 0,
    ) -> str:
        area = lookup_area(country_code)
        params = {
            "documentType": "A37",
            "businessType": "B75",
            "processType": "A47",
            "Acquiring_Domain": area.code,
            "offset": offset,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_bids_availability(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        business_type: str,
        offset: int = 0,
    ) -> str:
        """
        businessType
        [O] C40 = Conditional bid; C41 = Thermal limit; C42 = Frequency limit; C43 = Voltage limit;
        C44 = Current limit; C45 = Short-circuit current limits; C46 = Dynamic stability limit
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "B45",
            "businessType": business_type,
            "Domain": area.code,
            "processType": "A47",
            offset: offset,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_aggregated_bids(
        self, country_code: Union[Area, str], process_type: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> str:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        process_type : str
            A51 ... aFRR; A47 ... mFRR

        Returns
        -------
        str
        """
        if process_type not in ["A51", "A47"]:
            raise ValueError("processType allowed values: A51, A47")
        area = lookup_area(country_code)
        params = {"documentType": "A24", "area_Domain": area.code, "processType": process_type}
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_fcr_total_capacity(self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp) -> str:
        area = lookup_area(country_code)
        params = {
            "documentType": "A26",
            "businessType": "A25",
            "area_Domain": area.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_fcr_capacity_shares(self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp) -> str:
        area = lookup_area(country_code)
        params = {
            "documentType": "A26",
            "businessType": "C23",
            "area_Domain": area.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_fcr_sharing_between_sa(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp
    ) -> str:
        area = lookup_area(country_code)
        params = {
            "documentType": "A26",
            "businessType": "C22",
            "processType": "A52",
            "area_Domain": area.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_balancing_financial_expenses_and_income(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp
    ) -> str:
        area = lookup_area(country_code)
        params = {
            "documentType": "A87",
            "controlArea_Domain": area.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_total_imbalance_volumes(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp
    ) -> str:
        area = lookup_area(country_code)
        params = {
            "documentType": "A86",
            "controlArea_Domain": area.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_current_balancing_state(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp
    ) -> str:
        area = lookup_area(country_code)
        params = {
            "documentType": "A86",
            "businessType": "B33",
            "area_Domain": area.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_imbalance(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        process_type: str,
    ) -> str:
        area = lookup_area(country_code)
        params = {
            "documentType": "A45",
            "processType": process_type,
            "area_Domain": area.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_frr_rr_capacity_outlook(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, process_type: str
    ) -> str:
        area = lookup_area(country_code)
        params = {
            "documentType": "A26",
            "businessType": "C76",
            "processType": process_type,
            "area_Domain": area.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_frr_rr_actual_capacity(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        process_type: str,
        business_type: str,
    ) -> str:
        area = lookup_area(country_code)
        params = {
            "documentType": "A26",
            "businessType": business_type,
            "processType": process_type,
            "area_Domain": area.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_net_position(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, dayahead: bool = True
    ) -> str:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        dayahead : bool

        Returns
        -------
        str
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A25",  # Allocation result document
            "businessType": "B09",  # net position
            "Contract_MarketAgreement.Type": "A01",  # daily
            "in_Domain": area.code,
            "out_Domain": area.code,
        }
        if not dayahead:
            params.update({"Contract_MarketAgreement.Type": "A07"})

        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_load(self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp) -> str:
        """
        Parameters

        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        str
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A65",
            "processType": "A16",
            "outBiddingZone_Domain": area.code,
            "out_Domain": area.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_load_forecast(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, process_type: str = "A01"
    ) -> str:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        process_type : str

        Returns
        -------
        str
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A65",
            "processType": process_type,
            "outBiddingZone_Domain": area.code,
            # 'out_Domain': domain
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_flow_allocations(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, process_type: str = "A44"
    ) -> str:
        area = lookup_area(country_code)
        params = {
            "documentType": "B09",
            "processType": process_type,
            "domain.mRID": area.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_generation_forecast(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, process_type: str = "A01"
    ) -> str:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        process_type : str

        Returns
        -------
        str
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A71",
            "processType": process_type,
            "in_Domain": area.code,
        }
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_wind_and_solar_forecast(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Optional[str] = None,
        process_type: str = "A01",
        **kwargs,
    ) -> str:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter on a single psr type
        process_type : str

        Returns
        -------
        str
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A69",
            "processType": process_type,
            "in_Domain": area.code,
        }
        if psr_type:
            params.update({"psrType": psr_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_intraday_wind_and_solar_forecast(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, psr_type: Optional[str] = None
    ) -> str:
        return self.query_wind_and_solar_forecast(
            country_code=country_code, start=start, end=end, psr_type=psr_type, process_type="A40"
        )

    def query_generation(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter on a single psr type

        Returns
        -------
        str
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A75",
            "processType": "A16",
            "in_Domain": area.code,
        }
        if psr_type:
            params.update({"psrType": psr_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_generation_per_plant(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter on a single psr type

        Returns
        -------
        str
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A73",
            "processType": "A16",
            "in_Domain": area.code,
        }
        if psr_type:
            params.update({"psrType": psr_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_installed_generation_capacity(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, psr_type: Optional[str] = None
    ) -> str:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter query for a specific psr type

        Returns
        -------
        str
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A68",
            "processType": "A33",
            "in_Domain": area.code,
        }
        if psr_type:
            params.update({"psrType": psr_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_installed_generation_capacity_per_unit(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, psr_type: Optional[str] = None
    ) -> str:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter query for a specific psr type

        Returns
        -------
        str
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A71",
            "processType": "A33",
            "in_Domain": area.code,
        }
        if psr_type:
            params.update({"psrType": psr_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_aggregate_water_reservoirs_and_hydro_storage(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp
    ) -> str:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        offset : int
            offset for querying more than 100 documents

        Returns
        -------
        str
        """
        area = lookup_area(country_code)
        params = {"documentType": "A72", "processType": "A16", "in_Domain": area.code}
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_crossborder_flows(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> str:
        """
        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        str
        """
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A11",
            contract_marketagreement_type=None,
        )

    def query_scheduled_exchanges(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        dayahead: bool = False,
        **kwargs,
    ) -> str:
        """
        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        dayahead : bool
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        str
        """
        if dayahead:
            contract_marketagreement_type = "A01"
        else:
            contract_marketagreement_type = "A05"
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A09",
            contract_marketagreement_type=contract_marketagreement_type,
        )

    def query_dc_link_capacity_limits(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A93",
        )

    def query_congestion_costs(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, **kwargs
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code,
            country_code_to=country_code,
            start=start,
            end=end,
            doctype="A92",
            business_type="B04",
        )

    def query_countertrading_costs(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, **kwargs
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code,
            country_code_to=country_code,
            start=start,
            end=end,
            doctype="A92",
            business_type="B03",
        )

    def query_redispatching_costs(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, **kwargs
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code,
            country_code_to=country_code,
            start=start,
            end=end,
            doctype="A92",
            business_type="A46",
        )

    def query_redispatching_internal(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, **kwargs
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code,
            country_code_to=country_code,
            start=start,
            end=end,
            doctype="A63",
            business_type="A85",
        )

    def query_redispatching_crossborder(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A63",
            business_type="A46",
        )

    def query_expansion_dismantling_project(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A90",
            business_type="B01",
        )

    def query_countertrading(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A91",
        )

    def query_net_transfer_capacity_dayahead(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> str:
        """
        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        str
        """
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A61",
            contract_marketagreement_type="A01",
        )

    def query_net_transfer_capacity_weekahead(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> str:
        """
        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        str
        """
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A61",
            contract_marketagreement_type="A02",
        )

    def query_net_transfer_capacity_monthahead(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> str:
        """
        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        str
        """
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A61",
            contract_marketagreement_type="A03",
        )

    def query_net_transfer_capacity_yearahead(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> str:
        """
        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        str
        """
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A61",
            contract_marketagreement_type="A04",
        )

    def query_offered_capacity(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        contract_marketagreement_type: str,
        implicit: bool = True,
        **kwargs,
    ) -> str:
        """

        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        contract_marketagreement_type : str
            type of contract (see mappings.MARKETAGREEMENTTYPE)
        implicit: bool (True = implicit - default for most borders. False = explicit - for instance BE-GB)

        Returns
        -------
        str
        """
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A31",
            contract_marketagreement_type=contract_marketagreement_type,
            auction_type=("A01" if implicit else "A02"),
        )

    def query_capacity_usage(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        contract_marketagreement_type: str,
        **kwargs,
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A25",
            business_type="B05",
            auction_category="A04",
            contract_marketagreement_type=contract_marketagreement_type,
        )

    def query_energy_prices(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        contract_marketagreement_type: str,
        offset: int = 0,
        **kwargs,
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A44",
            contract_marketagreement_type=contract_marketagreement_type,
            offset=offset,
        )

    def query_auction_revenue(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        contract_marketagreement_type: str,
        **kwargs,
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A25",
            business_type="B07",
            auction_category="A04",
            contract_marketagreement_type=contract_marketagreement_type,
        )

    def query_congestion_income(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        contract_marketagreement_type: str,
        **kwargs,
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A25",
            contract_marketagreement_type=contract_marketagreement_type,
            business_type="B10",
        )

    def query_transfer_capacities_third_countries(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        contract_marketagreement_type: str,
        implicit: bool = True,
        **kwargs,
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A94",
            contract_marketagreement_type=contract_marketagreement_type,
            auction_type="A01" if implicit else "A02",
            auction_category="A04",
        )

    def query_total_capacity_allocated(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        contract_marketagreement_type: str,
        **kwargs,
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A26",
            contract_marketagreement_type=contract_marketagreement_type,
            business_type="A29",
            auction_category="A04",
        )

    def query_total_nominated_capacity(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A26",
            business_type="B08",
        )

    def query_balancing_border_capacity_limits(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        process_type: str,
        **kwargs,
    ) -> str:
        return self._query_crossborder(
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            doctype="A31",
            business_type="B26",
            process_type=process_type,
        )

    def _query_crossborder(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        doctype: str,
        contract_marketagreement_type: Optional[str] = None,
        auction_type: Optional[str] = None,
        auction_category: Optional[str] = None,
        business_type: Optional[str] = None,
        process_type: Optional[str] = None,
        offset: Optional[int] = None,
    ) -> str:
        """
        Generic function called by query_crossborder_flows,
        query_scheduled_exchanges, query_net_transfer_capacity_DA/WA/MA/YA and query_.

        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        doctype: str
        contract_marketagreement_type: str
        business_type: str

        Returns
        -------
        str
        """
        area_in = lookup_area(country_code_to)
        area_out = lookup_area(country_code_from)

        params = {"documentType": doctype, "in_Domain": area_in.code, "out_Domain": area_out.code}
        if contract_marketagreement_type is not None:
            params["contract_MarketAgreement.Type"] = contract_marketagreement_type
        if auction_type is not None:
            params["Auction.Type"] = auction_type
        if auction_category is not None:
            params["auction.Category"] = auction_category
        if business_type is not None:
            params["businessType"] = business_type
        if process_type is not None:
            params["processType"] = process_type
        if offset is not None:
            params["offset"] = offset

        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_activated_balancing_energy_prices(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        process_type: Optional[str] = "A16",
        psr_type: Optional[str] = None,
        business_type: Optional[str] = None,
        standard_market_product: Optional[str] = None,
        original_market_product: Optional[str] = None,
    ) -> bytes:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        process_type: str
            A16 used if not provided
        psr_type : str
            filter query for a specific psr type
        business_type: str
            filter query for a specific business type
        standard_market_product: str
        original_market_product: str
            filter query for a specific product
            defaults to standard product
        Returns
        -------
        bytes
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A84",
            "processType": process_type,
            "controlArea_Domain": area.code,
        }
        if psr_type:
            params.update({"psrType": psr_type})
        if business_type:
            params.update({"businessType": business_type})
        if standard_market_product:
            params.update({"standardMarketProduct": standard_market_product})
        if original_market_product:
            params.update({"originalMarketProduct": original_market_product})
        response = self._base_request(params=params, start=start, end=end)

        return response.content

    def query_afrr_marginal_prices(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Optional[str] = None,
    ) -> bytes:
        return self.query_activated_balancing_energy_prices(
            country_code=country_code,
            start=start,
            end=end,
            process_type="A67",
            psr_type=psr_type,
            business_type="A96",
            standard_market_product="A01",
        )

    def query_imbalance_prices(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, psr_type: Optional[str] = None
    ) -> bytes:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter query for a specific psr type

        Returns
        -------
        bytes
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A85",
            "controlArea_Domain": area.code,
        }
        if psr_type:
            params.update({"psrType": psr_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.content

    def query_accepted_aggregated_offers(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, psr_type: Optional[str] = None
    ) -> bytes:
        area = lookup_area(country_code)
        params = {
            "documentType": "A82",
            "controlArea_Domain": area.code,
        }
        if psr_type:
            params.update({"psrType": psr_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.content

    def query_imbalance_volumes(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, psr_type: Optional[str] = None
    ) -> bytes:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter query for a specific psr type

        Returns
        -------
        bytes
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A86",
            "controlArea_Domain": area.code,
        }
        if psr_type:
            params.update({"psrType": psr_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.content

    def query_procured_balancing_capacity(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        process_type: str,
        type_marketagreement_type: Optional[str] = None,
    ) -> bytes:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        process_type : str
            A51 ... aFRR; A47 ... mFRR
        type_marketagreement_type : str
            type of contract (see mappings.MARKETAGREEMENTTYPE)

        Returns
        -------
        bytes
        """
        if process_type not in ["A51", "A47"]:
            raise ValueError("processType allowed values: A51, A47")

        area = lookup_area(country_code)
        params = {"documentType": "A15", "area_Domain": area.code, "processType": process_type}
        if type_marketagreement_type:
            params.update({"type_MarketAgreement.Type": type_marketagreement_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.content

    def query_activated_balancing_energy(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        business_type: str,
        psr_type: Optional[str] = None,
    ) -> bytes:
        """
        Activated Balancing Energy [17.1.E]
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        business_type : str
            type of contract (see mappings.BSNTYPE)
        psr_type : str
            filter query for a specific psr type

        Returns
        -------
        bytes
        """
        area = lookup_area(country_code)
        params = {"documentType": "A83", "controlArea_Domain": area.code, "businessType": business_type}
        if psr_type:
            params.update({"psrType": psr_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.content

    def query_contracted_reserve_prices(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        type_marketagreement_type: str,
        psr_type: Optional[str] = None,
        offset: int = 0,
    ) -> str:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        type_marketagreement_type : str
            type of contract (see mappings.MARKETAGREEMENTTYPE)
        psr_type : str
            filter query for a specific psr type
        offset : int
            offset for querying more than 100 documents

        Returns
        -------
        str
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A89",
            "controlArea_Domain": area.code,
            "type_MarketAgreement.Type": type_marketagreement_type,
            "offset": offset,
        }
        if psr_type:
            params.update({"psrType": psr_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_balancing_reserve_prices(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        type_marketagreement_type: str,
        business_type: str,
        psr_type: Optional[str] = None,
        offset: int = 0,
    ) -> str:
        """
        businessType
        [O] A96 = Automatic frequency restoration reserve; A95 = Frequency Containment Reserve;
        A97 = Manual frequency restoration reserve; A98 = Replacement reserve
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A89",
            "businessType": business_type,
            "controlArea_Domain": area.code,
            "type_MarketAgreement.Type": type_marketagreement_type,
            "offset": offset,
        }
        if psr_type:
            params.update({"psrType": psr_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_contracted_reserve_prices_procured_capacity(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        process_type: str,
        type_marketagreement_type: str,
        psr_type: Optional[str] = None,
        offset: int = 0,
    ) -> str:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        process_type : str
            type of process (see mappings.PROCESSTYPE)
        type_marketagreement_type : str
            type of contract (see mappings.MARKETAGREEMENTTYPE)
        psr_type : str
            filter query for a specific psr type
        offset : int
            offset for querying more than 100 documents

        Returns
        -------
        str
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A81",  # [M] A81 = Contracted reserves
            "businessType": "B95",  # [M] B95 = Procured capacity
            "processType": process_type,
            # [M*] A51 = Automatic frequency restoration reserve; A52 =  Frequency containment reserve; A47 = Manual frequency restoration reserve; A46 = Replacement reserve
            "controlArea_Domain": area.code,
            "type_MarketAgreement.Type": type_marketagreement_type,
            "offset": offset,
        }
        if psr_type:
            params.update({"psrType": psr_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_balancing_reserve_under_contract(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        business_type: str,
        type_marketagreement_type: str,
        psr_type: Optional[str] = None,
        offset: int = 0,
    ) -> str:
        """
        Type_MarketAgreement.Type
        [M] A13 = Hourly; A01 = Daily; A02 = Weekly; A03 = Monthly; A04 = Yearly; A06 = Long Term

        businessType
        [O] A95 = Frequency containment reserve; A96 = Automatic frequency restoration reserve;
        A97 = Manual frequency restoration reserve; A98 = Replacement reserve
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A81",
            "businessType": business_type,
            "controlArea_Domain": area.code,
            "type_MarketAgreement.Type": type_marketagreement_type,
            "offset": offset,
        }
        if psr_type:
            params.update({"psrType": psr_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def query_contracted_reserve_amount(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        type_marketagreement_type: str,
        psr_type: Optional[str] = None,
        offset: int = 0,
    ) -> str:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        type_marketagreement_type : str
            type of contract (see mappings.MARKETAGREEMENTTYPE)
        psr_type : str
            filter query for a specific psr type
        offset : int
            offset for querying more than 100 documents

        Returns
        -------
        str
        """
        area = lookup_area(country_code)
        params = {
            "documentType": "A81",
            "controlArea_Domain": area.code,
            "type_MarketAgreement.Type": type_marketagreement_type,
            "offset": offset,
        }
        if psr_type:
            params.update({"psrType": psr_type})
        response = self._base_request(params=params, start=start, end=end)
        return response.text

    def _query_unavailability(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        doctype: str,
        docstatus: Optional[str] = None,
        periodstartupdate: Optional[pd.Timestamp] = None,
        periodendupdate: Optional[pd.Timestamp] = None,
        mRID=None,
        offset: int = 0,
    ) -> bytes:
        """
        Generic unavailibility query method.
        This endpoint serves ZIP files.
        The query is limited to 200 items per request.

        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        doctype : str
        docstatus : str, optional
        periodstartupdate : pd.Timestamp, optional
        periodendupdate : pd.Timestamp, optional
        offset : int

        Returns
        -------
        bytes
        """
        area = lookup_area(country_code)
        params = {
            "documentType": doctype,
            "biddingZone_domain": area.code,
            "offset": offset,
            # ,'businessType': 'A53 (unplanned) | A54 (planned)'
        }
        if docstatus:
            params["docStatus"] = docstatus
        if periodstartupdate and periodendupdate:
            params["periodStartUpdate"] = self._datetime_to_str(periodstartupdate)
            params["periodEndUpdate"] = self._datetime_to_str(periodendupdate)
        if mRID:
            params["mRID"] = mRID
        response = self._base_request(params=params, start=start, end=end)
        return response.content

    def query_unavailability_of_generation_units(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        docstatus: Optional[str] = None,
        periodstartupdate: Optional[pd.Timestamp] = None,
        periodendupdate: Optional[pd.Timestamp] = None,
        mRID=None,
        offset: int = 0,
    ) -> bytes:
        """
        This endpoint serves ZIP files.
        The query is limited to 200 items per request.

        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        docstatus : str, optional
        periodstartupdate : pd.Timestamp, optional
        periodendupdate : pd.Timestamp, optional
        offset : int


        Returns
        -------
        bytes
        """
        content = self._query_unavailability(
            country_code=country_code,
            start=start,
            end=end,
            doctype="A80",
            docstatus=docstatus,
            periodstartupdate=periodstartupdate,
            periodendupdate=periodendupdate,
            mRID=mRID,
            offset=offset,
        )
        return content

    def query_unavailability_of_consumption_units(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        docstatus: Optional[str] = None,
        periodstartupdate: Optional[pd.Timestamp] = None,
        periodendupdate: Optional[pd.Timestamp] = None,
        mRID=None,
        offset: int = 0,
    ) -> bytes:
        content = self._query_unavailability(
            country_code=country_code,
            start=start,
            end=end,
            doctype="A76",
            docstatus=docstatus,
            periodstartupdate=periodstartupdate,
            periodendupdate=periodendupdate,
            mRID=mRID,
            offset=offset,
        )
        return content

    def query_unavailability_of_offshore_grid(
        self, area_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp
    ):
        return self._query_unavailability(country_code=area_code, start=start, end=end, doctype="A79")

    def query_unavailability_of_production_units(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        docstatus: Optional[str] = None,
        periodstartupdate: Optional[pd.Timestamp] = None,
        periodendupdate: Optional[pd.Timestamp] = None,
        mRID: Optional[str] = None,
    ) -> bytes:
        """
        This endpoint serves ZIP files.
        The query is limited to 200 items per request.

        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        docstatus : str, optional
        periodstartupdate : pd.Timestamp, optional
        periodendupdate : pd.Timestamp, optional

        Returns
        -------
        bytes
        """
        content = self._query_unavailability(
            country_code=country_code,
            start=start,
            end=end,
            doctype="A77",
            docstatus=docstatus,
            periodstartupdate=periodstartupdate,
            periodendupdate=periodendupdate,
            mRID=mRID,
        )
        return content

    def query_unavailability_transmission(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        docstatus: Optional[str] = None,
        periodstartupdate: Optional[pd.Timestamp] = None,
        periodendupdate: Optional[pd.Timestamp] = None,
        offset: int = 0,
        **kwargs,
    ) -> bytes:
        """
        Generic unavailibility query method.
        This endpoint serves ZIP files.
        The query is limited to 200 items per request.

        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        docstatus : str, optional
        periodstartupdate : pd.Timestamp, optional
        periodendupdate : pd.Timestamp, optional
        offset : int

        Returns
        -------
        bytes
        """
        area_in = lookup_area(country_code_to)
        area_out = lookup_area(country_code_from)
        params = {"documentType": "A78", "in_Domain": area_in.code, "out_Domain": area_out.code, "offset": offset}
        if docstatus:
            params["docStatus"] = docstatus
        if periodstartupdate and periodendupdate:
            params["periodStartUpdate"] = self._datetime_to_str(periodstartupdate)
            params["periodEndUpdate"] = self._datetime_to_str(periodendupdate)
        response = self._base_request(params=params, start=start, end=end)
        return response.content

    def query_withdrawn_unavailability_of_generation_units(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, mRID: Optional[str] = None
    ) -> bytes:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        bytes
        """
        content = self._query_unavailability(
            country_code=country_code, start=start, end=end, doctype="A80", docstatus="A13", mRID=mRID
        )
        return content


class EntsoePandasClient(EntsoeRawClient):
    @year_limited
    def query_net_position(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        dayahead: bool = True,
        resolution: Literal["60min", "30min", "15min"] = "60min",
    ) -> pd.Series:
        """

        Parameters
        ----------
        country_code
        start
        end

        Returns
        -------

        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_net_position(
            country_code=area, start=start, end=end, dayahead=dayahead
        )
        series = parse_netpositions(text, resolution=resolution)
        if len(series) == 0:
            raise NoMatchingDataError
        series = series.tz_convert(area.tz)
        series = series.truncate(before=start, after=end)
        return series

    @year_limited
    def query_aggregated_bids(
        self, country_code: Union[Area, str], process_type: str, start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.Series:
        """

        Parameters
        ----------
        country_code
        start
        end
        process_type: str,

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_aggregated_bids(
            country_code=area, process_type=process_type, start=start, end=end
        )
        df = parse_aggregated_bids(text)
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    # we need to do offset, but we also want to pad the days so wrap it in an internal call
    def query_day_ahead_prices(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        resolution: Literal["60min", "30min", "15min"] = "60min",
    ) -> pd.Series:
        """
        Parameters
        ----------
        resolution: either 60min for hourly values,
            30min for half-hourly values or 15min for quarterly values, throws error if type is not available
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        pd.Series
        """
        if resolution not in ["60min", "30min", "15min"]:
            raise InvalidParameterError("Please choose either 60min, 30min or 15min")
        area = lookup_area(country_code)
        # we do here extra days at start and end to fix issue 187
        series = self._query_day_ahead_prices(
            area, start=start - pd.Timedelta(days=1), end=end + pd.Timedelta(days=1), resolution=resolution
        )
        series = series.tz_convert(area.tz).sort_index()
        series = series.truncate(before=start, after=end)
        # because of the above fix we need to check again if any valid data exists after truncating
        if len(series) == 0:
            raise NoMatchingDataError
        return series

    @year_limited
    @documents_limited(100)
    def _query_day_ahead_prices(
        self,
        area: Area,
        start: pd.Timestamp,
        end: pd.Timestamp,
        resolution: Literal["60min", "30min", "15min"] = "60min",
        offset: int = 0,
    ) -> pd.Series:
        text = super(EntsoePandasClient, self).query_day_ahead_prices(area, start=start, end=end, offset=offset)
        series = parse_prices(text)[resolution]

        if len(series) == 0:
            raise NoMatchingDataError
        return series

    @year_limited
    def query_load(self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_load(country_code=area, start=start, end=end)

        df = parse_loads(text, process_type="A16")
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    @year_limited
    def query_load_forecast(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, process_type: str = "A01"
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        process_type : str

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_load_forecast(
            country_code=area, start=start, end=end, process_type=process_type
        )

        df = parse_loads(text, process_type=process_type)
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    def _query_common_single_country(
        self,
        super_method: str,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        parse_func=parse_generic,
        **kwargs,
    ) -> pd.Series:
        area = lookup_area(country_code)
        method = getattr(super(EntsoePandasClient, self), super_method)
        text = method(country_code=area, start=start, end=end, **kwargs)
        df = parse_func(text)
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    def query_flow_allocations(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, process_type: str = "A44"
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_flow_allocations",
            country_code=country_code,
            start=start,
            end=end,
            process_type=process_type,
        )

    def query_load_and_forecast(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """
        utility function to combina query realised load and forecasted day ahead load.
        this mimics the html view on the page Total Load - Day Ahead / Actual

        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        pd.DataFrame
        """
        df_load_forecast_da = self.query_load_forecast(country_code, start=start, end=end)
        df_load = self.query_load(country_code, start=start, end=end)
        return df_load_forecast_da.join(df_load, sort=True, how="outer")

    @year_limited
    def query_generation_forecast(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        process_type: str = "A01",
        nett: bool = False,
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        process_type : str
        nett : bool
            condense generation and consumption into a nett number

        Returns
        -------
        pd.DataFrame | pd.Series
        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_generation_forecast(
            country_code=area, start=start, end=end, process_type=process_type
        )
        df = parse_generation(text, nett=nett)
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    @year_limited
    def query_wind_and_solar_forecast(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Optional[str] = None,
        process_type: str = "A01",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter on a single psr type
        process_type : str

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_wind_and_solar_forecast(
            country_code=area, start=start, end=end, psr_type=psr_type, process_type=process_type
        )
        df = parse_generation(text, nett=True)
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    def query_intraday_wind_and_solar_forecast(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, psr_type: Optional[str] = None
    ) -> pd.DataFrame:
        return self.query_wind_and_solar_forecast(
            country_code=country_code, start=start, end=end, psr_type=psr_type, process_type="A40"
        )

    @year_limited
    def query_generation(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Optional[str] = None,
        nett: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter on a single psr type
        nett : bool
            condense generation and consumption into a nett number

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_generation(
            country_code=area, start=start, end=end, psr_type=psr_type
        )
        df = parse_generation(text, nett=nett)
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    @year_limited
    def query_installed_generation_capacity(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, psr_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter query for a specific psr type

        Returns
        -------
        pd.DataFrame
        """

        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_installed_generation_capacity(
            country_code=area, start=start, end=end, psr_type=psr_type
        )
        df = parse_generation(text)
        df = df.tz_convert(area.tz)
        # Truncate to YearBegin and YearEnd, because answer is always year-based
        df = df.truncate(before=start - YearBegin(), after=end + YearEnd())
        return df

    @year_limited
    def query_installed_generation_capacity_per_unit(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, psr_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter query for a specific psr type

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_installed_generation_capacity_per_unit(
            country_code=area, start=start, end=end, psr_type=psr_type
        )
        df = parse_installed_capacity_per_plant(text)
        return df

    @year_limited
    @paginated
    def query_aggregate_water_reservoirs_and_hydro_storage(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_aggregate_water_reservoirs_and_hydro_storage(
            country_code=area, start=start, end=end
        )

        df = parse_water_hydro(text)

        return df

    @year_limited
    def query_crossborder_flows(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        """
        Note: Result will be in the timezone of the origin country

        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        pd.Series
        """
        area_to = lookup_area(country_code_to)
        area_from = lookup_area(country_code_from)
        text = super(EntsoePandasClient, self).query_crossborder_flows(
            country_code_from=area_from, country_code_to=area_to, start=start, end=end
        )
        ts = parse_crossborder_flows(text)
        ts = ts.tz_convert(area_from.tz)
        ts = ts.truncate(before=start, after=end)
        return ts

    @year_limited
    def query_scheduled_exchanges(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        dayahead: bool = False,
        **kwargs,
    ) -> pd.Series:
        """
        Note: Result will be in the timezone of the origin country

        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        dayahead : bool
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        pd.Series
        """
        area_to = lookup_area(country_code_to)
        area_from = lookup_area(country_code_from)
        text = super(EntsoePandasClient, self).query_scheduled_exchanges(
            country_code_from=area_from, country_code_to=area_to, dayahead=dayahead, start=start, end=end
        )
        ts = parse_crossborder_flows(text)
        ts = ts.tz_convert(area_from.tz)
        ts = ts.truncate(before=start, after=end)
        return ts

    @year_limited
    def query_net_transfer_capacity_dayahead(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        """
        Note: Result will be in the timezone of the origin country

        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        pd.Series
        """
        area_to = lookup_area(country_code_to)
        area_from = lookup_area(country_code_from)
        text = super(EntsoePandasClient, self).query_net_transfer_capacity_dayahead(
            country_code_from=area_from, country_code_to=area_to, start=start, end=end
        )
        ts = parse_crossborder_flows(text)
        ts = ts.tz_convert(area_from.tz)
        ts = ts.truncate(before=start, after=end)
        return ts

    @year_limited
    def query_net_transfer_capacity_weekahead(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        """
        Note: Result will be in the timezone of the origin country

        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        pd.Series
        """
        area_to = lookup_area(country_code_to)
        area_from = lookup_area(country_code_from)
        text = super(EntsoePandasClient, self).query_net_transfer_capacity_weekahead(
            country_code_from=area_from, country_code_to=area_to, start=start, end=end
        )
        ts = parse_crossborder_flows(text)
        ts = ts.tz_convert(area_from.tz)
        ts = ts.truncate(before=start, after=end)
        return ts

    @year_limited
    def query_net_transfer_capacity_monthahead(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        """
        Note: Result will be in the timezone of the origin country

        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        pd.Series
        """
        area_to = lookup_area(country_code_to)
        area_from = lookup_area(country_code_from)
        text = super(EntsoePandasClient, self).query_net_transfer_capacity_monthahead(
            country_code_from=area_from, country_code_to=area_to, start=start, end=end
        )
        ts = parse_crossborder_flows(text)
        ts = ts.tz_convert(area_from.tz)
        ts = ts.truncate(before=start, after=end)
        return ts

    @year_limited
    def query_net_transfer_capacity_yearahead(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        """
        Note: Result will be in the timezone of the origin country

        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        pd.Series
        """
        area_to = lookup_area(country_code_to)
        area_from = lookup_area(country_code_from)
        text = super(EntsoePandasClient, self).query_net_transfer_capacity_yearahead(
            country_code_from=area_from, country_code_to=area_to, start=start, end=end
        )
        ts = parse_crossborder_flows(text)
        ts = ts.tz_convert(area_from.tz)
        ts = ts.truncate(before=start, after=end)
        return ts

    @year_limited
    @paginated
    # @documents_limited(100)
    def query_offered_capacity(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        contract_marketagreement_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        implicit: bool = True,
        offset: int = 0,
        **kwargs,
    ) -> pd.Series:
        """
        Note: Result will be in the timezone of the origin country  --> to check

        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        contract_marketagreement_type : str
            type of contract (see mappings.MARKETAGREEMENTTYPE)
        start : pd.Timestamp
        end : pd.Timestamp
        implicit: bool (True = implicit - default for most borders. False = explicit - for instance BE-GB)
        offset: int
            offset for querying more than 100 documents
        Returns
        -------
        pd.Series
        """
        area_to = lookup_area(country_code_to)
        area_from = lookup_area(country_code_from)
        text = super(EntsoePandasClient, self).query_offered_capacity(
            country_code_from=area_from,
            country_code_to=area_to,
            start=start,
            end=end,
            contract_marketagreement_type=contract_marketagreement_type,
            implicit=implicit,
            offset=offset,
        )
        ts = parse_crossborder_flows(text)
        ts = ts.tz_convert(area_from.tz)
        ts = ts.truncate(before=start, after=end)
        return ts

    def _query_common_crossborder(
        self,
        super_method: str,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        parse_function=parse_crossborder_flows,
        **kwargs,
    ) -> pd.Series:
        area_to = lookup_area(country_code_to)
        area_from = lookup_area(country_code_from)
        method = getattr(super(EntsoePandasClient, self), super_method)
        text = method(country_code_from=area_from, country_code_to=area_to, start=start, end=end, **kwargs)
        ts = parse_function(text)
        ts = ts.tz_convert(area_from.tz)
        ts = ts.truncate(before=start, after=end)
        return ts

    @year_limited
    @paginated
    @documents_limited(100)
    def query_energy_prices(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        contract_marketagreement_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        offset: int = 0,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_energy_prices",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            contract_marketagreement_type=contract_marketagreement_type,
            offset=offset,
        )

    @paginated
    def query_redispatching_crossborder(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_redispatching_crossborder",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
        )

    @paginated
    def query_frr_rr_sharing(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_frr_rr_sharing",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
        )

    @paginated
    def query_exchanged_reserve_capacity(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        offset: int = 0,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_exchanged_reserve_capacity",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            offset=offset,
            start=start,
            end=end,
        )

    @paginated
    def query_cross_zonal_balancing_capacity_allocation_and_use(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        type_marketagreement_type: str,
        process_type: str,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_cross_zonal_balancing_capacity_allocation_and_use",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            type_marketagreement_type=type_marketagreement_type,
            process_type=process_type,
            start=start,
            end=end,
        )

    @paginated
    def query_crossborder_balancing(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_crossborder_balancing",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
        )

    @paginated
    def query_netted_and_exchanged_volumes(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        process_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_netted_and_exchanged_volumes",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            process_type=process_type,
        )

    @paginated
    def query_elastic_demands(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, **kwargs
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_elastic_demands",
            country_code=country_code,
            start=start,
            end=end,
        )

    @paginated
    def query_fcr_total_capacity(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, **kwargs
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_fcr_total_capacity",
            country_code=country_code,
            start=start,
            end=end,
        )

    @paginated
    def query_fcr_capacity_shares(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, **kwargs
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_fcr_capacity_shares",
            country_code=country_code,
            start=start,
            end=end,
        )

    @paginated
    def query_frr_rr_capacity_outlook(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, process_type: str, **kwargs
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_frr_rr_capacity_outlook",
            country_code=country_code,
            process_type=process_type,
            start=start,
            end=end,
        )

    @paginated
    def query_frr_rr_actual_capacity(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        process_type: str,
        business_type: str,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_frr_rr_actual_capacity",
            country_code=country_code,
            process_type=process_type,
            business_type=business_type,
            start=start,
            end=end,
        )

    @paginated
    def query_balancing_financial_expenses_and_income(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, **kwargs
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_balancing_financial_expenses_and_income",
            country_code=country_code,
            start=start,
            end=end,
        )

    @paginated
    def query_total_imbalance_volumes(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, **kwargs
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_total_imbalance_volumes",
            country_code=country_code,
            start=start,
            end=end,
        )

    @paginated
    def query_current_balancing_state(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, **kwargs
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_current_balancing_state",
            country_code=country_code,
            start=start,
            end=end,
        )

    @paginated
    def query_imbalance(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, process_type: str, **kwargs
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_imbalance",
            country_code=country_code,
            process_type=process_type,
            start=start,
            end=end,
        )

    @paginated
    def query_fcr_sharing_between_sa(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, **kwargs
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_fcr_sharing_between_sa",
            country_code=country_code,
            start=start,
            end=end,
        )

    @paginated
    def query_balancing_energy_bids(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        process_type: str,
        offset: int = 0,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_balancing_energy_bids",
            country_code=country_code,
            process_type=process_type,
            offset=offset,
            start=start,
            end=end,
        )

    @paginated
    def query_bids_availability(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        business_type: str,
        offset: int = 0,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_bids_availability",
            country_code=country_code,
            business_type=business_type,
            offset=offset,
            start=start,
            end=end,
        )

    @paginated
    def query_capacity_usage(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        contract_marketagreement_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        offset: int = 0,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_capacity_usage",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            contract_marketagreement_type=contract_marketagreement_type,
            offset=offset,
        )

    @paginated
    def query_expansion_dismantling_project(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_expansion_dismantling_project",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
        )

    @paginated
    def query_countertrading(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_countertrading",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
        )

    @paginated
    def query_auction_revenue(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        contract_marketagreement_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        offset: int = 0,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_auction_revenue",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            contract_marketagreement_type=contract_marketagreement_type,
            offset=offset,
        )

    @paginated
    def query_congestion_income(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        contract_marketagreement_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        offset: int = 0,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_congestion_income",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            contract_marketagreement_type=contract_marketagreement_type,
            offset=offset,
        )

    @paginated
    def query_transfer_capacities_third_countries(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        contract_marketagreement_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        offset: int = 0,
        implicit: bool = True,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_transfer_capacities_third_countries",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            contract_marketagreement_type=contract_marketagreement_type,
            offset=offset,
            implicit=implicit,
        )

    @paginated
    def query_total_capacity_allocated(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        contract_marketagreement_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        offset: int = 0,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_total_capacity_allocated",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            contract_marketagreement_type=contract_marketagreement_type,
            offset=offset,
        )

    @paginated
    def query_total_nominated_capacity(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        offset: int = 0,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_total_nominated_capacity",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
            offset=offset,
        )

    @paginated
    def query_balancing_border_capacity_limits(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        process_type: str,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_balancing_border_capacity_limits",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            process_type=process_type,
            start=start,
            end=end,
        )

    @paginated
    def query_dc_link_capacity_limits(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_crossborder(
            super_method="query_dc_link_capacity_limits",
            country_code_from=country_code_from,
            country_code_to=country_code_to,
            start=start,
            end=end,
        )

    @paginated
    def query_congestion_costs(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_congestion_costs",
            country_code=country_code,
            start=start,
            end=end,
        )

    @paginated
    def query_accepted_aggregated_offers(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_accepted_aggregated_offers",
            country_code=country_code,
            start=start,
            end=end,
        )

    def query_countertrading_costs(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_countertrading_costs",
            country_code=country_code,
            start=start,
            end=end,
        )

    def query_redispatching_costs(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_redispatching_costs",
            country_code=country_code,
            start=start,
            end=end,
        )

    def query_redispatching_internal(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        **kwargs,
    ) -> pd.Series:
        return self._query_common_single_country(
            super_method="query_redispatching_internal",
            country_code=country_code,
            start=start,
            end=end,
        )

    @year_limited
    def query_activated_balancing_energy_prices(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        process_type: Optional[str] = "A16",
        psr_type: Optional[str] = None,
        business_type: Optional[str] = None,
        standard_market_product: Optional[str] = None,
        original_market_product: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        process_type: str
            A16 used if not provided
        psr_type : str
            filter query for a specific psr type
        business_type: str
            filter query for a specific business type
        standard_market_product: str
        original_market_product: str
            filter query for a specific product
            defaults to standard product
        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_activated_balancing_energy_prices(
            country_code=area,
            start=start,
            end=end,
            process_type=process_type,
            psr_type=psr_type,
            business_type=business_type,
            standard_market_product=standard_market_product,
            original_market_product=original_market_product,
        )
        df = parse_activated_balancing_energy_prices(text)
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    @year_limited
    def query_afrr_marginal_prices(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Optional[str] = None,
    ) -> pd.DataFrame:
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_afrr_marginal_prices(
            country_code=area,
            start=start,
            end=end,
            psr_type=psr_type,
        )
        df = parse_activated_balancing_energy_prices(text)
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    @year_limited
    def query_imbalance_prices(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, psr_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter query for a specific psr type

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        archive = super(EntsoePandasClient, self).query_imbalance_prices(
            country_code=area, start=start, end=end, psr_type=psr_type
        )
        df = parse_imbalance_prices_zip(zip_contents=archive)
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    @year_limited
    def query_imbalance_volumes(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, psr_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter query for a specific psr type

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        archive = super(EntsoePandasClient, self).query_imbalance_volumes(
            country_code=area, start=start, end=end, psr_type=psr_type
        )
        df = parse_imbalance_volumes_zip(zip_contents=archive)
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    @year_limited
    @paginated
    def query_procured_balancing_capacity(
        self,
        country_code: Union[Area, str],
        process_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        type_marketagreement_type: Optional[str] = None,
    ) -> bytes:
        """
        Parameters
        ----------
        country_code : Area|str
        process_type : str
            A51 ... aFRR; A47 ... mFRR
        start : pd.Timestamp
        end : pd.Timestamp
        type_marketagreement_type : str
            type of contract (see mappings.MARKETAGREEMENTTYPE)

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_procured_balancing_capacity(
            country_code=area,
            start=start,
            end=end,
            process_type=process_type,
            type_marketagreement_type=type_marketagreement_type,
        )
        df = parse_procured_balancing_capacity(text, area.tz)
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    @year_limited
    def query_activated_balancing_energy(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        business_type: str,
        psr_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Activated Balancing Energy [17.1.E]
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        business_type: str
            type of contract (see mappings.BSNTYPE)
        psr_type : str
            filter query for a specific psr type

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_activated_balancing_energy(
            country_code=area, start=start, end=end, business_type=business_type, psr_type=psr_type
        )
        df = parse_contracted_reserve(text, area.tz, "quantity")
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    @year_limited
    @paginated
    @documents_limited(100)
    def query_contracted_reserve_prices(
        self,
        country_code: Union[Area, str],
        type_marketagreement_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Optional[str] = None,
        offset: int = 0,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area, str
        type_marketagreement_type : str
            type of contract (see mappings.MARKETAGREEMENTTYPE)
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter query for a specific psr type
        offset : int
            offset for querying more than 100 documents

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_contracted_reserve_prices(
            country_code=area,
            start=start,
            end=end,
            type_marketagreement_type=type_marketagreement_type,
            psr_type=psr_type,
            offset=offset,
        )
        df = parse_contracted_reserve(text, area.tz, "procurement_price.amount")
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    @year_limited
    @paginated
    @documents_limited(100)
    def query_contracted_reserve_prices_procured_capacity(
        self,
        country_code: Union[Area, str],
        process_type: str,
        type_marketagreement_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Optional[str] = None,
        offset: int = 0,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area, str
        process_type : str
            type of process (see mappings.PROCESSTYPE)
        type_marketagreement_type : str
            type of contract (see mappings.MARKETAGREEMENTTYPE)
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter query for a specific psr type
        offset : int
            offset for querying more than 100 documents

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_contracted_reserve_prices_procured_capacity(
            country_code=area,
            start=start,
            end=end,
            process_type=process_type,
            type_marketagreement_type=type_marketagreement_type,
            psr_type=psr_type,
            offset=offset,
        )
        df = parse_contracted_reserve(text, area.tz, "procurement_price.amount")
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    @paginated
    def query_balancing_reserve_under_contract(
        self,
        country_code: Union[Area, str],
        business_type: str,
        type_marketagreement_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Optional[str] = None,
        offset: int = 0,
    ) -> pd.DataFrame:

        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_balancing_reserve_under_contract(
            country_code=area,
            start=start,
            end=end,
            business_type=business_type,
            type_marketagreement_type=type_marketagreement_type,
            psr_type=psr_type,
            offset=offset,
        )
        df = parse_contracted_reserve(text, area.tz, "amount")
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    @paginated
    def query_balancing_reserve_prices(
        self,
        country_code: Union[Area, str],
        business_type: str,
        type_marketagreement_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Optional[str] = None,
        offset: int = 0,
    ) -> pd.DataFrame:

        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_balancing_reserve_prices(
            country_code=area,
            start=start,
            end=end,
            business_type=business_type,
            type_marketagreement_type=type_marketagreement_type,
            psr_type=psr_type,
            offset=offset,
        )
        df = parse_contracted_reserve(text, area.tz, "amount")
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    @year_limited
    @paginated
    @documents_limited(100)
    def query_contracted_reserve_amount(
        self,
        country_code: Union[Area, str],
        type_marketagreement_type: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Optional[str] = None,
        offset: int = 0,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        type_marketagreement_type : str
            type of contract (see mappings.MARKETAGREEMENTTYPE)
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter query for a specific psr type
        offset : int
            offset for querying more than 100 documents

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_contracted_reserve_amount(
            country_code=area,
            start=start,
            end=end,
            type_marketagreement_type=type_marketagreement_type,
            psr_type=psr_type,
            offset=offset,
        )
        df = parse_contracted_reserve(text, area.tz, "quantity")
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        return df

    @year_limited
    @paginated
    @documents_limited(200)
    def _query_unavailability(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        doctype: str,
        docstatus: Optional[str] = None,
        periodstartupdate: Optional[pd.Timestamp] = None,
        periodendupdate: Optional[pd.Timestamp] = None,
        mRID: Optional[str] = None,
        offset: int = 0,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        doctype : str
        docstatus : str, optional
        periodstartupdate : pd.Timestamp, optional
        periodendupdate : pd.Timestamp, optional
        offset : int

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        content = super(EntsoePandasClient, self)._query_unavailability(
            country_code=area,
            start=start,
            end=end,
            doctype=doctype,
            docstatus=docstatus,
            periodstartupdate=periodstartupdate,
            periodendupdate=periodendupdate,
            mRID=mRID,
            offset=offset,
        )
        df = parse_unavailabilities(content, doctype)
        df = df.tz_convert(area.tz)
        df["start"] = df["start"].apply(lambda x: x.tz_convert(area.tz))
        df["end"] = df["end"].apply(lambda x: x.tz_convert(area.tz))
        df = df[(df["start"] < end) | (df["end"] > start)]
        return df

    def query_unavailability_of_offshore_grid(
        self, area_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        zipfile = super(EntsoePandasClient, self)._query_unavailability(
            country_code=area_code, start=start, end=end, doctype="A79"
        )
        df = parse_offshore_unavailability(zipfile)
        return df

    def query_unavailability_of_generation_units(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        docstatus: Optional[str] = None,
        periodstartupdate: Optional[pd.Timestamp] = None,
        periodendupdate: Optional[pd.Timestamp] = None,
        mRID=None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        docstatus : str, optional
        periodstartupdate : pd.Timestamp, optional
        periodendupdate : pd.Timestamp, optional

        Returns
        -------
        pd.DataFrame
        """
        df = self._query_unavailability(
            country_code=country_code,
            start=start,
            end=end,
            doctype="A80",
            docstatus=docstatus,
            periodstartupdate=periodstartupdate,
            periodendupdate=periodendupdate,
            mRID=mRID,
        )
        return df

    def query_unavailability_of_consumption_units(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        docstatus: Optional[str] = None,
        periodstartupdate: Optional[pd.Timestamp] = None,
        periodendupdate: Optional[pd.Timestamp] = None,
        mRID=None,
        **kwargs,
    ) -> pd.DataFrame:
        df = self._query_unavailability(
            country_code=country_code,
            start=start,
            end=end,
            doctype="A76",
            docstatus=docstatus,
            periodstartupdate=periodstartupdate,
            periodendupdate=periodendupdate,
            mRID=mRID,
        )
        return df

    def query_unavailability_of_production_units(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        docstatus: Optional[str] = None,
        periodstartupdate: Optional[pd.Timestamp] = None,
        periodendupdate: Optional[pd.Timestamp] = None,
        mRID: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        docstatus : str, optional
        periodstartupdate : pd.Timestamp, optional
        periodendupdate : pd.Timestamp, optional

        Returns
        -------
        pd.DataFrame
        """
        df = self._query_unavailability(
            country_code=country_code,
            start=start,
            end=end,
            doctype="A77",
            docstatus=docstatus,
            periodstartupdate=periodstartupdate,
            periodendupdate=periodendupdate,
            mRID=mRID,
        )
        return df

    @paginated
    def query_unavailability_transmission(
        self,
        country_code_from: Union[Area, str],
        country_code_to: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        docstatus: Optional[str] = None,
        periodstartupdate: Optional[pd.Timestamp] = None,
        periodendupdate: Optional[pd.Timestamp] = None,
        offset: int = 0,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code_from : Area|str
        country_code_to : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        docstatus : str, optional
        periodstartupdate : pd.Timestamp, optional
        periodendupdate : pd.Timestamp, optional
        offset : int

        Returns
        -------
        pd.DataFrame
        """
        area_to = lookup_area(country_code_to)
        area_from = lookup_area(country_code_from)
        content = super(EntsoePandasClient, self).query_unavailability_transmission(
            area_from, area_to, start, end, docstatus, periodstartupdate, periodendupdate, offset=offset
        )
        df = parse_unavailabilities(content, "A78")
        df = df.tz_convert(area_from.tz)
        df["start"] = df["start"].apply(lambda x: x.tz_convert(area_from.tz))
        df["end"] = df["end"].apply(lambda x: x.tz_convert(area_from.tz))
        df = df[(df["start"] < end) | (df["end"] > start)]
        return df

    def query_withdrawn_unavailability_of_generation_units(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp, **kwargs
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp

        Returns
        -------
        pd.DataFrame
        """
        df = self.query_unavailability_of_generation_units(
            country_code=country_code, start=start, end=end, docstatus="A13"
        )
        df = df[(df["start"] < end) | (df["end"] > start)]
        return df

    @day_limited
    def query_generation_per_plant(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        psr_type: Optional[str] = None,
        include_eic: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        country_code : Area|str
        start : pd.Timestamp
        end : pd.Timestamp
        psr_type : str
            filter on a single psr type
        include_eic: bool
            if True also include the eic code in the output

        Returns
        -------
        pd.DataFrame
        """
        area = lookup_area(country_code)
        text = super(EntsoePandasClient, self).query_generation_per_plant(
            country_code=area, start=start, end=end, psr_type=psr_type
        )
        df = parse_generation(text, per_plant=True, include_eic=include_eic)
        df.columns = df.columns.set_levels(df.columns.levels[0].str.encode("latin-1").str.decode("utf-8"), level=0)
        df = df.tz_convert(area.tz)
        # Truncation will fail if data is not sorted along the index in rare
        # cases. Ensure the dataframe is sorted:
        df = df.sort_index(axis=0)

        if df.columns.nlevels == 2:
            df = df.assign(newlevel="Actual Aggregated").set_index("newlevel", append=True).unstack("newlevel")
        df = df.truncate(before=start, after=end)
        return df

    def query_physical_crossborder_allborders(
        self,
        country_code: Union[Area, str],
        start: pd.Timestamp,
        end: pd.Timestamp,
        export: bool,
        per_hour: bool = False,
    ) -> pd.DataFrame:
        """
        Adds together all physical cross-border flows to a country for a given direction
        The neighbours of a country are given by the NEIGHBOURS mapping

        if export is True then all export flows are returned, if False then all import flows are returned
        some borders have more then once data points per hour. Set per_hour=True if you always want hourly data,
        it will then thake the mean
        """
        area = lookup_area(country_code)
        imports = []
        for neighbour in NEIGHBOURS[area.name]:
            try:
                if export:
                    im = self.query_crossborder_flows(
                        country_code_from=country_code,
                        country_code_to=neighbour,
                        end=end,
                        start=start,
                        lookup_bzones=True,
                    )
                else:
                    im = self.query_crossborder_flows(
                        country_code_from=neighbour,
                        country_code_to=country_code,
                        end=end,
                        start=start,
                        lookup_bzones=True,
                    )
            except NoMatchingDataError:
                continue
            im.name = neighbour
            imports.append(im)
        df = pd.concat(imports, axis=1, sort=True)
        # drop columns that contain only zero's
        df = df.loc[:, (df != 0).any(axis=0)]
        df = df.tz_convert(area.tz)
        df = df.truncate(before=start, after=end)
        df["sum"] = df.sum(axis=1)
        if per_hour:
            df = df.resample("h").first()

        return df

    def query_import(self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """
        Utility function wrapper for query_sum_physical_crossborder for backwards compatibility reason
        """
        return self.query_physical_crossborder_allborders(country_code=country_code, start=start, end=end, export=False)

    def query_generation_import(
        self, country_code: Union[Area, str], start: pd.Timestamp, end: pd.Timestamp
    ) -> pd.DataFrame:
        """Query the combination of both domestic generation and imports"""
        generation = self.query_generation(country_code=country_code, end=end, start=start, nett=True)
        generation = generation.loc[:, (generation != 0).any(axis=0)]  # drop columns that contain only zero's
        imports = self.query_import(country_code=country_code, start=start, end=end)

        data = {f"Generation": generation, f"Import": imports}
        df = pd.concat(data.values(), axis=1, keys=data.keys())
        df = df.ffill()
        df = df.truncate(before=start, after=end)
        return df
