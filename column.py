from enum import Enum

class Column(Enum):
    # index from the data
    index = 1

    # order id for each transaction, should be unique across all row
    order_id = 2

    # when order is placed
    date = 3

    # status of the order detailed, simplified version is in status_courier
    status = 4

    # which party fulfill the order, currently only Amazon and non Amazon
    fulfilment = 5

    # Where the order is placed, most of the time is Amazon
    sales_channel = 6

    # What kind of shipment this order use,
    # expedited means faster arrival time compared to regular
    ship_service_level = 7

    # Style ID of the item, most item here are fashion and apparel
    style = 8

    # Stock keeping unit is the primary identifier of an item
    # same SKU should be same item
    stock_keeping_unit = 9 # SKU

    # Category of the item
    category = 10

    # Size of the item purchased in fashion sense. S, M, L, etc
    size = 11

    # Similar to SKU but for internal Amazon
    amazon_item_id = 12 # ASIN

    # Shipment status of the order, simpler version of status column
    courier_status = 13

    # Quantity a transaction purchase. This is what we want to forecast since it
    # represent demand by customer
    quantity = 14

    # Currency of the transaction. Currently only exists in india rupee
    currency = 15

    # total amount customer need to pay for the item
    amount = 16

    # City destination of the shipment. Having inconsistencies. Need further cleaning
    ship_city = 17

    # State of the shipment destination
    ship_state = 18

    # Postal code of the shipment destination
    ship_postal_code = 19

    # country of the shipment destination
    ship_country = 20

    # which promo code customer use in the transaction
    promotion_id = 21

    # is it for business to business purchase
    b2b = 22

    # the main vendor. for unknown reason, there is only one vendor
    fulfilled_by = 23

    # unknown column
    unknown = 24

    # ---- Synthetic Data
    # week when this transaction happen
    week = 25

    # month when this transaction happen
    month = 26

    # indicates that the purchase use some promotional code
    is_promotion = 27

    @classmethod
    def column_name(cls):
        return [col.name for col in cls]

    @classmethod
    def new_index(cls):
        return cls.order_id.name

    @classmethod
    def target(cls):
        return cls.quantity.name

    @classmethod
    def item_identifier(cls):
        return [
            cls.stock_keeping_unit.name,
            cls.amazon_item_id.name,
        ]

    # see EDA section for better explanation
    @classmethod
    def dropped(cls):
        return [
            cls.index.name,
            cls.status.name,
            cls.sales_channel.name,
            cls.style.name,
            cls.currency.name,
            cls.ship_country.name,
            cls.promotion_id.name,
            cls.b2b.name,
            cls.fulfilled_by.name,
            cls.stock_keeping_unit.name,
            cls.amazon_item_id.name,
            cls.amount.name,
            cls.ship_city.name,
            cls.ship_postal_code.name,
            cls.unknown.name
        ]

    @classmethod
    def categorical(cls):
        return [
            cls.fulfilment.name,
            cls.ship_service_level.name,
            cls.category.name,
            cls.size.name,
            cls.courier_status.name,
            cls.ship_state.name,
        ]

    @classmethod
    def time(cls):
        return [
            cls.date.name,
        ]

    @classmethod
    def numerical(cls):
        return []

    @classmethod
    def require_ohe(cls):
        return [
            cls.fulfilment.name,
            cls.ship_service_level.name,
            cls.category.name,
            cls.size.name,
            cls.courier_status.name,
            cls.ship_state.name,
            cls.week.name,
            cls.month.name,
        ]