from app import db
from sqlalchemy import Column, Integer, Float, DateTime, String


class BTCPrice(db.Model):
    __tablename__ = 'btcprices'

    btcprice_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False)
    base = Column(String(25), nullable=False)
    currency = Column(String(25), nullable=False)
    buy = Column(Float, nullable=False, default=-1.0)
    sell = Column(Float, nullable=False, default=-1.0)
    spot = Column(Float, nullable=False, default=-1.0)
    cmc = Column(Float, nullable=False, default=-1.0)

    def __init__(self, ts, base, cur, buy, sell, spot, cmc):
        self.timestamp = ts
        self.base = base
        self.currency = cur
        self.buy = buy
        self.sell = sell
        self.spot = spot
        self.cmc = cmc

    def __repr__(self):
        return '<id {}>'.format(self.id)
