import bs4
import pandas as pd


def _extract_timeseries(xml_text, label="timeseries"):
    """
    Parameters
    ----------
    xml_text : str

    Yields
    -------
    bs4.element.tag
    """
    if not xml_text:
        return
    soup = bs4.BeautifulSoup(xml_text, 'html.parser')
    for timeseries in soup.find_all(label):
        yield timeseries


RESOLUTIONS = {
    'PT60M': '60min',
    'P1Y': '12MS',
    'PT15M': '15min',
    'PT30M': '30min',
    'P1D': '1D',
    'P7D': '7D',
    'P1M': '1MS',
    'PT1M': '1min',
    'PT4S': '4D 4h',
}


def _resolution_to_timedelta(res_text: str) -> str:
    """
    Convert an Entsoe resolution to something that pandas can understand
    """

    delta = RESOLUTIONS.get(res_text)
    if delta is None:
        raise NotImplementedError("Sorry, I don't know what to do with the "
                                  "resolution '{}', because there was no "
                                  "documentation to be found of this format. "
                                  "Everything is hard coded. Please open an "
                                  "issue.".format(res_text))
    return delta


def _parse_datetimeindex(soup, tz=None):
    """
    Create a datetimeindex from a parsed beautifulsoup,
    given that it contains the elements 'start', 'end'
    and 'resolution'

    Parameters
    ----------
    soup : bs4.element.tag
    tz: str

    Returns
    -------
    pd.DatetimeIndex
    """
    start = pd.Timestamp(soup.find('start').text)
    end = pd.Timestamp(soup.find_all('end')[-1].text)
    if tz is not None:
        start = start.tz_convert(tz)
        end = end.tz_convert(tz)

    delta = _resolution_to_timedelta(res_text=soup.find('resolution').text)
    index = pd.date_range(start=start, end=end, freq=delta, inclusive='left')
    if tz is not None:
        dst_jump = len(set(index.map(lambda d: d.dst()))) > 1
        if dst_jump and delta == "7D":
            # For a weekly granularity, if we jump over the DST date in October,
            # date_range erronously returns an additional index element
            # because that week contains 169 hours instead of 168.
            index = index[:-1]
        index = index.tz_convert("UTC")
    elif index.to_series().diff().min() >= pd.Timedelta('1D') and end.hour == start.hour + 1:
        # For a daily or larger granularity, if we jump over the DST date in October,
        # date_range erronously returns an additional index element
        # because the period contains one extra hour.
        index = index[:-1]

    return index


def _parse_timeseries_generic(
    soup, label='quantity', label2=None, merge_series=False, period_name='period', add_classification_sequence=False, add_direction=False, add_contract=False
):
    series = {resolution: [] for resolution in RESOLUTIONS.values()}

    for period in soup.find_all(period_name):
        data = {}
        start = pd.Timestamp(period.find('start').text)
        end = pd.Timestamp(period.find('end').text)
        delta_text = _resolution_to_timedelta(res_text=period.find('resolution').text)
        delta = pd.Timedelta(delta_text)

        for point in period.find_all('point'):
            value = (point.find(label) or point.find(label2)).text
            value = value.replace(',', '')
            position = int(point.find('position').text)
            data[start + (position - 1) * delta] = value

        if len(data) == 1 and list(data.values())[0] == '0':
            continue

        S = pd.Series(data).sort_index()
        if soup.find('curvetype').text == 'A03':
            # with A03 its possible that positions are missing, this is when values are repeated
            # see docs: https://eepublicdownloads.entsoe.eu/clean-documents/EDI/Library/cim_based/Introduction_of_different_Timeseries_possibilities__curvetypes__with_ENTSO-E_electronic_document_v1.4.pdf
            # so lets do reindex on a continious range which creates gaps if positions are missing
            # then forward fill, so repeat last valid value, to fill the gaps
            S = S.reindex(pd.date_range(start, end - delta, freq=delta_text)).ffill()

        if add_classification_sequence or add_direction or add_contract:
            S = pd.DataFrame(S, columns=["value"])
        if (
            add_classification_sequence and
            (classification_sequence := soup.find(lambda tag: tag.name.lower() == 'classificationsequence_attributeinstancecomponent.position')) and
            classification_sequence.text
        ):
            S["classification_sequence"] = classification_sequence.text
        if (
            add_direction and
            (direction := soup.find(lambda tag: tag.name.lower() == 'flowdirection.direction')) and
            direction.text
        ):
            S["direction"] = direction.text
        if (
            add_contract and
            (contract := soup.find(lambda tag: tag.name.lower() == 'contract_marketagreement.type')) and
            contract.text
        ):
            S["contract_marketagreement_type"] = contract.text

        if delta_text not in series:
            series[delta_text] = []
        series[delta_text].append(S)

    for freq, S in series.items():
        if len(S) > 0:
            series[freq] = pd.concat(S).sort_index()
            if isinstance(series[freq], pd.Series):
                series[freq] = series[freq].astype(float)
            else:
                series[freq]["value"] = series[freq]["value"].astype(float)
        else:
            series[freq] = None

    # for endpoints which never has duplicated timeseries the flag merge_series signals to just concat everything
    if merge_series:
        all_series = [val for val in series.values() if val is not None]
        return pd.concat(all_series) if all_series else pd.DataFrame()
    else:
        return series


def _parse_timeseries_generic_whole(xml_text, label='quantity', label2=None, ts_label="timeseries", take_last=False, **kwargs):
    series_all = []
    for soup in _extract_timeseries(xml_text, label=ts_label):
        series_all.append(_parse_timeseries_generic(soup, label=label, label2=label2, merge_series=True, **kwargs))

    if take_last:
        series_all = series_all[-1].sort_index()
    else:
        series_all = pd.concat(series_all).sort_index()
    return series_all
